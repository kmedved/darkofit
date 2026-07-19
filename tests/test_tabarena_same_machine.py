"""Mutation-oriented tests for the frozen same-machine comparator campaign."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


if sys.version_info < (3, 10):
    pytest.skip(
        "standalone benchmark tooling requires Python 3.10+",
        allow_module_level=True,
    )

from benchmarks import analyze_tabarena_regression_same_machine as analysis
from benchmarks import run_tabarena_regression_same_machine as campaign


BENCHMARK_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("autogluon", "catboost", "tabarena")
)
requires_benchmark_deps = pytest.mark.skipif(
    not BENCHMARK_DEPS_AVAILABLE,
    reason="TabArena, AutoGluon, and CatBoost are not all installed",
)
if BENCHMARK_DEPS_AVAILABLE:
    from benchmarks import tabarena_comparator_adapters as adapters
    from benchmarks.tabarena_screen_adapters import (
        AIRFOIL_COLUMNS,
        DIAMONDS_CHILD_CODE_RANKS,
        DIAMONDS_COLUMNS,
        DIAMONDS_ORDERS,
    )
else:
    # These placeholders make dependency-gated parametrization collectable;
    # none is executed without the optional benchmark stack.
    adapters = None
    AIRFOIL_COLUMNS = ()
    DIAMONDS_COLUMNS = ()
    DIAMONDS_ORDERS = {"cut": (), "color": (), "clarity": ()}
    DIAMONDS_CHILD_CODE_RANKS = {
        "cut": tuple(range(5)),
        "color": tuple(range(7)),
        "clarity": tuple(range(8)),
    }


EXPECTED_TASKS = {
    "airfoil_self_noise": 363612,
    "Another-Dataset-on-used-Fiat-500": 363615,
    "concrete_compressive_strength": 363625,
    "diamonds": 363631,
    "Food_Delivery_Time": 363672,
    "healthcare_insurance_expenses": 363675,
    "houses": 363678,
    "miami_housing": 363686,
    "physiochemical_protein": 363693,
    "QSAR-TID-11": 363697,
    "QSAR_fish_toxicity": 363698,
    "superconductivity": 363705,
    "wine_quality": 363708,
}

EXPECTED_CYCLE = (
    ("D", "M", "C"),
    ("M", "C", "D"),
    ("C", "D", "M"),
    ("C", "M", "D"),
    ("D", "C", "M"),
    ("M", "D", "C"),
)


def _fake_jobs() -> list[SimpleNamespace]:
    experiments = {}
    for arm, spec in campaign.ARM_SPECS.items():
        model_cls = type(spec["model_cls"], (), {})
        experiments[arm] = SimpleNamespace(
            name=campaign._experiment_name(arm),
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": {
                    "ag_args": {
                        "name_suffix": campaign._experiment_suffix(
                            spec["lane"], spec["code"]
                        )
                    },
                    "ag_args_ensemble": campaign.expected_ag_ensemble_config(),
                },
                "fit_kwargs": {"num_cpus": 18},
            },
        )
    return [
        SimpleNamespace(
            experiment=experiments[arm],
            task=SimpleNamespace(dataset=dataset, repeat=repeat, fold=fold),
        )
        for lane, dataset, repeat, fold, arm in reversed(
            campaign.expected_ordered_grid()
        )
    ]


def _instantiate(model_cls):
    model = model_cls(
        path="",
        name=f"{model_cls.__name__}Probe",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={},
    )
    model.initialize()
    return model


def _airfoil_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frequency": [800.0, 1_600.0, 2_500.0],
            "chord-length": [0.30, 0.20, 0.10],
            "free-stream-velocity": [31.7, 39.6, 55.5],
            "suction-side-displacement-thickness": [0.002, 0.004, 0.006],
            "attack-angle": pd.Categorical(
                [0, 13, 26], categories=range(27)
            ),
        },
        columns=AIRFOIL_COLUMNS,
    )


def _diamonds_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "carat": [0.5, 1.0, 1.5],
            "depth": [61.0, 62.0, 63.0],
            "table": [55.0, 56.0, 57.0],
            "x": [4.0, 5.0, 6.0],
            "y": [4.1, 5.1, 6.1],
            "z": [2.5, 3.1, 3.8],
            "cut": pd.Categorical([0, 2, 4], categories=range(5)),
            "color": pd.Categorical([0, 3, 6], categories=range(7)),
            "clarity": pd.Categorical([0, 4, 7], categories=range(8)),
        },
        columns=DIAMONDS_COLUMNS,
    )
    return frame


def _ordinal_outputs(frame: pd.DataFrame):
    results = []
    for model_cls in (
        adapters.ComparatorOrdinalDarkoFitModel,
        adapters.ComparatorOrdinalChimeraBoostModel,
        adapters.ComparatorOrdinalCatBoostModel,
    ):
        model = _instantiate(model_cls)
        model._fit_representation(frame)
        transformed, unknown_count = model._transform_representation(frame)
        results.append(
            (transformed, unknown_count, model._representation_metadata())
        )
    return results


def _valid_comparator_fit(engine: str, *, unknown_external_stop: bool = False):
    requested = 1_000 if engine == "darkofit" else 10_000
    common = {
        "schema_version": 1,
        "engine": engine,
        "iterations_requested": requested,
        "best_iteration": 80,
        "rounds_retained": 80,
        "iterations_attempted": 90,
        "resolved_params": {"thread_count": 18},
        "num_cpus": 18,
        "num_gpus": 0,
        "stop_reason": (
            None if unknown_external_stop and engine != "darkofit" else "early_stopping"
        ),
    }
    if engine == "darkofit":
        common.update(
            {
                "resolved_learning_rate": 0.07,
                "requested_tree_mode": "catboost",
                "selected_tree_mode": "catboost",
                "selected_lane": "boosting",
                "deadline_hit": False,
            }
        )
    elif engine == "chimeraboost":
        common.update(
            {
                "resolved_learning_rate": 0.1,
                "selected_lane": "constant",
                "linear_leaves_selected": False,
                "linear_selection_performed": True,
                "stop_reason_inferred": not unknown_external_stop,
            }
        )
    else:
        common["resolved_params"]["task_type"] = "CPU"
        common.update(
            {
                "resolved_learning_rate": 0.05,
                "tree_count": 80,
                "catboost_best_iteration_zero_based": 79,
                "stop_reason_inferred": not unknown_external_stop,
            }
        )
    return common


def _valid_analyzer_comparator_fit(engine: str, *, unknown_stop: bool = False):
    fit = _valid_comparator_fit(
        engine, unknown_external_stop=unknown_stop and engine != "darkofit"
    )
    if engine == "darkofit":
        fit.update(
            {
                "rounds_completed": 85,
                "wall_clock_limit_seconds": 100.0,
                "wall_clock_safety_margin_seconds": 5.0,
                "wall_clock_effective_seconds": 95.0,
                "wall_clock_elapsed_seconds": 10.0,
                "deadline_is_soft": True,
            }
        )
    return fit


def _minimal_outer_rows() -> list[dict]:
    rows = []
    for lane, dataset, repeat, fold, arm in campaign.expected_ordered_grid():
        spec = campaign.ARM_SPECS[arm]
        # Positive deterministic values are sufficient for pairing; using a
        # code-specific multiplier makes every contrast nontrivial.
        multiplier = {"D": 1.0, "M": 2.0, "C": 4.0}[spec["code"]]
        rows.append(
            {
                "lane": lane,
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "arm": arm,
                "engine": spec["engine"],
                **{
                    metric: multiplier * (1.0 + repeat + fold / 10.0)
                    for metric in analysis.METRICS
                },
            }
        )
    return rows


def test_frozen_panel_lane_counts_and_resources_are_exact():
    assert campaign.TASKS == EXPECTED_TASKS
    assert campaign.PRIMARY_COORDINATE_PAIRS == ((0, 0), (1, 1), (2, 2))
    assert campaign.DIAGNOSTIC_DATASETS == ("airfoil_self_noise", "diamonds")
    assert campaign.EXPECTED_PRIMARY_COORDINATES == 39
    assert campaign.EXPECTED_DIAGNOSTIC_COORDINATES == 6
    assert campaign.EXPECTED_PRIMARY_JOBS == 117
    assert campaign.EXPECTED_DIAGNOSTIC_JOBS == 18
    assert campaign.EXPECTED_JOBS == 135
    assert campaign.EXPECTED_CHILD_FITS == 1_080
    assert len(campaign.expected_grid()) == 135

    assert campaign.expected_ag_ensemble_config() == {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": 3_600.0,
    }
    assert campaign.expected_fit_kwargs_extra(18) == {
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "raise_on_model_failure": True,
        "calibrate": False,
        "num_cpus": 18,
    }
    for invalid in (17, 19, True, 18.5, "18"):
        with pytest.raises(RuntimeError):
            campaign.expected_fit_kwargs_extra(invalid)
    assert campaign.parse_args(["--time-limit", "3600", "--dry-run"]).dry_run
    with pytest.raises(SystemExit):
        campaign.parse_args(["--time-limit", "3599"])
    with pytest.raises(SystemExit):
        campaign.parse_args(["--dry-run", "--resume"])


def test_primary_cycle_is_continuous_and_diagnostic_cycle_is_separate():
    assert campaign.ORDER_CYCLE == EXPECTED_CYCLE
    ordered = campaign.expected_ordered_grid()
    groups = [ordered[index : index + 3] for index in range(0, len(ordered), 3)]
    primary_groups = groups[:39]
    diagnostic_groups = groups[39:]

    for index, group in enumerate(primary_groups):
        assert len({row[:4] for row in group}) == 1
        assert [campaign.ARM_SPECS[row[4]]["code"] for row in group] == list(
            EXPECTED_CYCLE[index % 6]
        )
    # Thirteen datasets times three coordinates means Diamonds does not get a
    # per-dataset cycle restart; its first primary group is cycle row four.
    diamonds_group_index = list(EXPECTED_TASKS).index("diamonds") * 3
    assert [
        campaign.ARM_SPECS[row[4]]["code"]
        for row in primary_groups[diamonds_group_index]
    ] == list(EXPECTED_CYCLE[diamonds_group_index % 6])

    assert len(diagnostic_groups) == 6
    for index, group in enumerate(diagnostic_groups):
        assert [campaign.ARM_SPECS[row[4]]["code"] for row in group] == list(
            EXPECTED_CYCLE[index]
        )
        assert all(row[0] == campaign.ORDINAL_DIAGNOSTIC_LANE for row in group)

    audit = campaign.ordering_audit(campaign.order_comparator_jobs(_fake_jobs()))
    assert audit["lane_position_counts"] == {
        "primary": {
            engine: {"first": 13, "second": 13, "third": 13}
            for engine in ("darkofit", "chimeraboost", "catboost")
        },
        "ordinal_diagnostic": {
            engine: {"first": 2, "second": 2, "third": 2}
            for engine in ("darkofit", "chimeraboost", "catboost")
        },
    }


def test_runner_and_opaque_analyzer_reconstruct_the_same_order_digest():
    assert analysis.expected_ordered_grid() == campaign.expected_ordered_grid()
    assert analysis.job_order_sha256() == campaign.job_order_sha256()
    expected_payload = [
        {
            "lane": lane,
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
            "arm_code": campaign.ARM_SPECS[arm]["code"],
        }
        for lane, dataset, repeat, fold, arm in campaign.expected_ordered_grid()
    ]
    digest = hashlib.sha256(
        campaign.hardened._canonical_json(expected_payload)
    ).hexdigest()
    assert campaign.job_order_sha256() == digest


def test_job_grid_mutations_are_rejected():
    ordered = campaign.order_comparator_jobs(_fake_jobs())

    with pytest.raises(
        RuntimeError,
        match="frozen comparator coordinates|does not contain all engines",
    ):
        campaign.order_comparator_jobs(ordered[:-1])

    duplicated = ordered[:-1] + [ordered[0]]
    with pytest.raises(RuntimeError, match="duplicate"):
        campaign.order_comparator_jobs(duplicated)

    misordered = ordered.copy()
    misordered[0], misordered[1] = misordered[1], misordered[0]
    with pytest.raises(RuntimeError, match="frozen cycle"):
        campaign.ordering_audit(misordered)

    broken = deepcopy(ordered[0])
    broken.experiment.method_kwargs["model_hyperparameters"]["leaked"] = 1
    with pytest.raises(RuntimeError, match="frozen arm"):
        campaign._job_arm(broken)


def test_protocol_separates_native_characterization_from_ordinal_diagnostic():
    protocol = campaign.frozen_protocol()
    assert protocol["expected_jobs"] == 135
    assert protocol["expected_child_fits"] == 1_080
    assert protocol["order_cycle_codes"] == [list(row) for row in EXPECTED_CYCLE]
    assert protocol["order_cycle_scope"] == {
        "primary": "continuous_across_all_39_coordinate_groups",
        "ordinal_diagnostic": "one_complete_six_group_cycle",
    }
    assert protocol["lanes"]["primary"]["pool_with_other_lanes"] is False
    diagnostic = protocol["lanes"]["ordinal_diagnostic"]
    assert diagnostic["pool_with_other_lanes"] is False
    assert diagnostic["policy_advancement_allowed"] is False
    assert protocol["analysis"] == {
        "primary_dataset_weighting": "equal_1_over_13",
        "ordinal_diagnostic_dataset_weighting": "equal_1_over_2",
        "estimator": "paired_log_ratio",
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20_260_719,
        "bootstrap_resampling": "coordinates_within_each_fixed_dataset",
        "lanes_pooled": False,
    }
    assert all(
        spec["manual_config"] == {} and spec["official_defaults"] is True
        for spec in protocol["arms"].values()
    )
    assert protocol["chimera_source"] == {
        "version": "0.14.1",
        "exact_git_commit": "9c9ea6e704a9fe2bfe6d6c284b22de73914be048",
        "hidden_import_warmup": "disabled",
    }
    assert protocol["catboost_source"] == {"version": "1.2.10"}
    assert all(
        dispatch["register_results_in_context"] is False
        for dispatch in protocol["execution_dispatch"]
    )
    assert protocol["competitor_stop_reason_policy"] == {
        "known_time_or_deadline_invalidates": True,
        "unresolved_external_callback_outcome_allowed": True,
        "unresolved_is_reported_not_inferred": True,
        "qualification": (
            "may_include_early_stopping_time_memory_or_no_legal_split"
        ),
    }


@requires_benchmark_deps
def test_native_and_ordinal_adapters_preserve_each_engine_official_defaults():
    pairs = {
        "darkofit": (
            adapters.ComparatorDarkoFitModel,
            adapters.ComparatorOrdinalDarkoFitModel,
            {
                "iterations": 1_000,
                "early_stopping": True,
                "tree_mode": "catboost",
                "diagnostic_warnings": "never",
                "random_state": 0,
            },
        ),
        "chimeraboost": (
            adapters.ComparatorChimeraBoostModel,
            adapters.ComparatorOrdinalChimeraBoostModel,
            {"n_estimators": 10_000, "early_stopping": True, "random_state": 0},
        ),
        "catboost": (
            adapters.ComparatorCatBoostModel,
            adapters.ComparatorOrdinalCatBoostModel,
            {
                "iterations": 10_000,
                "learning_rate": 0.05,
                "allow_writing_files": False,
                "eval_metric": "RMSE",
                "random_seed": 0,
            },
        ),
    }
    for native_cls, ordinal_cls, expected in pairs.values():
        native = _instantiate(native_cls)._get_model_params()
        ordinal = _instantiate(ordinal_cls)._get_model_params()
        assert native == ordinal == expected

    loaded = campaign._load_model_classes()
    assert set(loaded) == {
        cls.__name__
        for pair in pairs.values()
        for cls in pair[:2]
    }
    for spec in campaign.ARM_SPECS.values():
        assert spec["config"] == {}
        assert loaded[spec["model_cls"]].__module__ == (
            "benchmarks.tabarena_comparator_adapters"
        )


@pytest.mark.parametrize(
    ("frame_factory", "encoded_columns", "expected_values"),
    [
        (
            _airfoil_frame,
            ("attack-angle",),
            {"attack-angle": [0.0, 3.0, 9.9]},
        ),
        (
            _diamonds_frame,
            tuple(DIAMONDS_ORDERS),
            {
                "cut": [
                    DIAMONDS_CHILD_CODE_RANKS["cut"][index]
                    for index in (0, 2, 4)
                ],
                "color": [
                    DIAMONDS_CHILD_CODE_RANKS["color"][index]
                    for index in (0, 3, 6)
                ],
                "clarity": [
                    DIAMONDS_CHILD_CODE_RANKS["clarity"][index]
                    for index in (0, 4, 7)
                ],
            },
        ),
    ],
)
@requires_benchmark_deps
def test_all_three_ordinal_adapters_apply_identical_frozen_numeric_values(
    frame_factory, encoded_columns, expected_values
):
    outputs = _ordinal_outputs(frame_factory())
    reference_frame, reference_unknown, reference_metadata = outputs[0]
    assert reference_unknown == 0
    for column in encoded_columns:
        assert reference_frame[column].dtype == np.dtype("float64")
        assert reference_frame[column].tolist() == expected_values[column]
    for transformed, unknown_count, metadata in outputs[1:]:
        assert_frame_equal(transformed, reference_frame, check_exact=True)
        assert unknown_count == reference_unknown
        assert metadata == reference_metadata

    assert reference_metadata["mapping_source"] == "source_frozen_before_campaign"
    assert reference_metadata["remaining_native_target_stat_positions"] == []
    assert reference_metadata["unknown_policy"] == "fail_closed"
    assert len(reference_metadata["category_schema_sha256"]) == 64


@requires_benchmark_deps
def test_ordinal_adapters_fail_closed_on_schema_and_compact_domain_mutations():
    frame = _airfoil_frame()
    model = _instantiate(adapters.ComparatorOrdinalDarkoFitModel)
    with pytest.raises(RuntimeError, match="predeclared schema"):
        model._fit_representation(frame[list(reversed(frame.columns))])

    changed_domain = frame.copy()
    changed_domain["attack-angle"] = pd.Categorical(
        [0, 13, 25], categories=range(26)
    )
    with pytest.raises(RuntimeError, match="compact category domain changed"):
        model._fit_representation(changed_domain)

    model._fit_representation(frame)
    renamed = frame.rename(columns={"frequency": "frequency_mutated"})
    with pytest.raises(RuntimeError, match="input schema changed"):
        model._transform_representation(renamed)


def test_common_comparator_fit_schema_accepts_truthful_unknown_external_stop():
    for arm, spec in campaign.ARM_SPECS.items():
        unknown = spec["engine"] in {"chimeraboost", "catboost"}
        fitted = _valid_comparator_fit(
            spec["engine"], unknown_external_stop=unknown
        )
        normalized = campaign._validate_comparator_fit(
            fitted, arm=arm, field=f"{arm}.comparator_fit"
        )
        assert normalized["engine"] == spec["engine"]
        if unknown:
            assert normalized["stop_reason"] is None
            assert normalized["stop_reason_inferred"] is False
            assert "deadline_hit" not in normalized
        else:
            assert normalized["stop_reason"] == "early_stopping"
            assert normalized["deadline_hit"] is False


def test_opaque_analyzer_enforces_exact_common_and_engine_telemetry_schemas():
    for engine in ("darkofit", "chimeraboost", "catboost"):
        fit = _valid_analyzer_comparator_fit(
            engine, unknown_stop=engine != "darkofit"
        )
        normalized = analysis._validate_common_comparator_fit(
            fit, engine=engine, child_cpus=18, field=f"{engine} telemetry"
        )
        assert normalized == fit
        if engine != "darkofit":
            assert normalized["stop_reason"] is None
            assert normalized["stop_reason_inferred"] is False

        extra = dict(fit, undeclared_field=True)
        with pytest.raises(RuntimeError, match="fields are not exact"):
            analysis._validate_common_comparator_fit(
                extra, engine=engine, child_cpus=18, field=f"{engine} extra"
            )


def test_runtime_provenance_is_exact_and_mutation_sensitive():
    recorded = campaign.collect_runtime_provenance()
    analysis._verify_runtime_provenance(recorded)
    for mutate in (
        lambda value: value.__setitem__("python_version", "0.0.0"),
        lambda value: value["packages"].__setitem__("catboost", "0.0.0"),
        lambda value: value["environment"].__setitem__(
            "CHIMERABOOST_WARMUP", "mutated"
        ),
        lambda value: value["hardware"].__setitem__(
            "host_identity_sha256", "0" * 64
        ),
    ):
        changed = deepcopy(recorded)
        mutate(changed)
        with pytest.raises(RuntimeError, match="runtime/hardware"):
            analysis._verify_runtime_provenance(changed)


@requires_benchmark_deps
def test_catboost_positional_fit_resources_follow_the_official_argument_order():
    from autogluon.tabular.models import CatBoostModel

    parameters = list(inspect.signature(CatBoostModel._fit).parameters)
    assert parameters[:8] == [
        "self",
        "X",
        "y",
        "X_val",
        "y_val",
        "time_limit",
        "num_gpus",
        "num_cpus",
    ]

    class FakeCatBoostCore:
        tree_count_ = 80

        @staticmethod
        def get_all_params():
            return {"learning_rate": 0.05}

        @staticmethod
        def get_best_iteration():
            return 79

        @staticmethod
        def get_evals_result():
            return {"validation": {"RMSE": [1.0] * 80}}

    class FakeCatBoostBase:
        def __init__(self):
            self._fit_metadata = {}
            self.model = FakeCatBoostCore()

        @staticmethod
        def _get_model_params():
            return {"iterations": 10_000}

        def _fit(self, X, y, *args, **kwargs):
            return None

    class Probe(adapters._CatBoostTelemetryMixin, FakeCatBoostBase):
        pass

    probe = Probe()
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0.0, 1.0])
    # Official positional order after X/y is X_val, y_val, time_limit,
    # num_gpus, num_cpus. Telemetry must not reverse the final two values.
    probe._fit(X, y, None, None, 100.0, 0, 18)
    metadata = probe._fit_metadata[adapters.COMPARATOR_METADATA_KEY]
    assert metadata["num_cpus"] == 18
    assert metadata["num_gpus"] == 0


@requires_benchmark_deps
def test_chimeraboost_stop_reason_is_unresolved_for_timed_auto_selection():
    infer = adapters._infer_chimera_stop_reason
    assert infer(
        attempted=90,
        retained=80,
        requested=10_000,
        time_limit=3_600.0,
        linear_selection_performed=True,
    ) is None
    assert infer(
        attempted=10_000,
        retained=10_000,
        requested=10_000,
        time_limit=3_600.0,
        linear_selection_performed=True,
    ) is None
    assert infer(
        attempted=90,
        retained=80,
        requested=10_000,
        time_limit=3_600.0,
        linear_selection_performed=False,
    ) == "early_stopping"
    assert infer(
        attempted=90,
        retained=80,
        requested=10_000,
        time_limit=None,
        linear_selection_performed=True,
    ) == "early_stopping"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda fit: fit.pop("resolved_params"), "common telemetry"),
        (lambda fit: fit.__setitem__("iterations_requested", 999), "iteration"),
        (lambda fit: fit.__setitem__("num_cpus", 17), "resources"),
        (lambda fit: fit.__setitem__("num_gpus", 1), "resources"),
        (lambda fit: fit.__setitem__("deadline_hit", True), "deadline|wall-clock"),
        (lambda fit: fit.__setitem__("stop_reason", "time_limit"), "deadline|wall-clock"),
    ],
)
def test_common_comparator_fit_schema_rejects_execution_mutations(
    mutation, expected_error
):
    arm = "darkofit_product_default"
    fitted = _valid_comparator_fit("darkofit")
    mutation(fitted)
    with pytest.raises(RuntimeError, match=expected_error):
        campaign._validate_comparator_fit(
            fitted, arm=arm, field="mutated comparator fit"
        )


def test_child_configs_are_engine_specific_and_fold_seeded():
    assert campaign.expected_child_hyperparameters("darkofit", 7) == {
        "iterations": 1_000,
        "early_stopping": True,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
        "random_state": 7,
    }
    assert campaign.expected_child_hyperparameters("chimeraboost", 7) == {
        "n_estimators": 10_000,
        "early_stopping": True,
        "random_state": 7,
    }
    assert campaign.expected_child_hyperparameters("catboost", 7) == {
        "iterations": 10_000,
        "learning_rate": 0.05,
        "allow_writing_files": False,
        "eval_metric": "RMSE",
        "random_seed": 7,
    }
    for engine in campaign.ENGINE_SPECS:
        for invalid in (-1, 8, True, 1.0, "0"):
            with pytest.raises(RuntimeError, match="child_fold"):
                campaign.expected_child_hyperparameters(engine, invalid)


def test_chimeraboost_source_activation_requires_exact_tag_clean_import_identity(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "chimera"
    package = checkout / "chimeraboost"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "0.14.1"\n', encoding="utf-8")
    fake_module = SimpleNamespace(__file__=str(module_path), __version__="0.14.1")

    def clean_git(args, *, cwd):
        assert cwd == checkout
        command = tuple(args)
        return {
            ("rev-parse", "HEAD"): campaign.CHIMERABOOST_TAG_COMMIT,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("tag", "--points-at", "HEAD"): "v0.14.1",
            ("rev-parse", "HEAD^{tree}"): "a" * 40,
            ("remote", "get-url", "origin"): "https://github.com/kmedved/chimeraboost.git",
        }[command]

    monkeypatch.setattr(campaign, "_run_git", clean_git)
    monkeypatch.setattr(campaign.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(
        campaign.importlib,
        "import_module",
        lambda name: fake_module if name == "chimeraboost" else None,
    )
    monkeypatch.delenv("CHIMERABOOST_WARMUP", raising=False)
    old_path = list(sys.path)
    try:
        provenance = campaign.activate_chimeraboost_checkout(checkout)
    finally:
        sys.path[:] = old_path
    assert provenance["git_head"] == campaign.CHIMERABOOST_TAG_COMMIT
    assert provenance["git_tag"] == "v0.14.1"
    assert provenance["git_remote_origin"] == (
        "https://github.com/kmedved/chimeraboost.git"
    )
    assert provenance["module_file"] == str(module_path.resolve())
    assert provenance["module_sha256"] == hashlib.sha256(
        module_path.read_bytes()
    ).hexdigest()


def test_chimeraboost_activation_replaces_a_preloaded_foreign_package(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "chimera"
    package = checkout / "chimeraboost"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "0.14.1"\n', encoding="utf-8")
    foreign = tmp_path / "foreign/chimeraboost"
    foreign.mkdir(parents=True)
    foreign_root = SimpleNamespace(
        __file__=str(foreign / "__init__.py"),
        __version__="0.15.0",
    )
    foreign_tree = SimpleNamespace(__file__=str(foreign / "tree.py"))

    def clean_git(args, *, cwd):
        assert cwd == checkout
        return {
            ("rev-parse", "HEAD"): campaign.CHIMERABOOST_TAG_COMMIT,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("tag", "--points-at", "HEAD"): "v0.14.1",
            ("rev-parse", "HEAD^{tree}"): "a" * 40,
            ("remote", "get-url", "origin"): (
                "https://github.com/kmedved/chimeraboost.git"
            ),
        }[tuple(args)]

    monkeypatch.setattr(campaign, "_run_git", clean_git)
    monkeypatch.delenv("CHIMERABOOST_WARMUP", raising=False)
    monkeypatch.setitem(sys.modules, "chimeraboost", foreign_root)
    monkeypatch.setitem(sys.modules, "chimeraboost.tree", foreign_tree)
    old_path = list(sys.path)
    try:
        provenance = campaign.activate_chimeraboost_checkout(checkout)
        loaded = sys.modules["chimeraboost"]
        assert loaded is not foreign_root
        assert Path(loaded.__file__).resolve() == module_path.resolve()
        assert "chimeraboost.tree" not in sys.modules
    finally:
        sys.path[:] = old_path

    assert provenance["module_file"] == str(module_path.resolve())


def test_chimeraboost_activation_rejects_an_already_mixed_namespace(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "chimera"
    package = checkout / "chimeraboost"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "0.14.1"\n', encoding="utf-8")
    expected_root = SimpleNamespace(
        __file__=str(module_path),
        __version__="0.14.1",
    )
    foreign_tree = SimpleNamespace(
        __file__=str(tmp_path / "foreign/chimeraboost/tree.py")
    )

    def clean_git(args, *, cwd):
        assert cwd == checkout
        return {
            ("rev-parse", "HEAD"): campaign.CHIMERABOOST_TAG_COMMIT,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("tag", "--points-at", "HEAD"): "v0.14.1",
        }[tuple(args)]

    monkeypatch.setattr(campaign, "_run_git", clean_git)
    monkeypatch.delenv("CHIMERABOOST_WARMUP", raising=False)
    monkeypatch.setitem(sys.modules, "chimeraboost", expected_root)
    monkeypatch.setitem(sys.modules, "chimeraboost.tree", foreign_tree)

    with pytest.raises(RuntimeError, match="mixed ChimeraBoost imports"):
        campaign.activate_chimeraboost_checkout(checkout)

    assert sys.modules["chimeraboost"] is expected_root
    assert sys.modules["chimeraboost.tree"] is foreign_tree


@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
def test_chimeraboost_hidden_warmup_accepts_only_actual_disabled_values(value):
    assert campaign._validate_chimeraboost_warmup_environment(value) == value


@pytest.mark.parametrize("value", [" ", "false", "no", "off", "1", "background"])
def test_chimeraboost_hidden_warmup_rejects_truthy_upstream_values(value):
    with pytest.raises(RuntimeError, match="unset, empty, or zero"):
        campaign._validate_chimeraboost_warmup_environment(value)


@requires_benchmark_deps
def test_warmup_history_uses_the_same_hidden_warmup_contract(monkeypatch):
    from benchmarks import tabarena_comparator_warmup as warmup

    for value in ("", "0", " 0 "):
        monkeypatch.setenv("CHIMERABOOST_WARMUP", value)
        assert warmup._warmup_environment_value() == value
    for value in (" ", "false", "no", "off", "1", "background"):
        monkeypatch.setenv("CHIMERABOOST_WARMUP", value)
        with pytest.raises(RuntimeError, match="unset, empty, or zero"):
            warmup._warmup_environment_value()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("head", "exact v0.14.1 commit"),
        ("dirty", "not clean"),
        ("tag", "not tagged"),
        ("warmup", "must be unset, empty, or zero"),
        ("version", "version is not 0.14.1"),
        ("module", "not the frozen checkout"),
    ],
)
def test_chimeraboost_source_activation_rejects_provenance_mutations(
    tmp_path, monkeypatch, mutation, expected_error
):
    checkout = tmp_path / "chimera"
    package = checkout / "chimeraboost"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "0.14.1"\n', encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")

    def mutated_git(args, *, cwd):
        command = tuple(args)
        values = {
            ("rev-parse", "HEAD"): campaign.CHIMERABOOST_TAG_COMMIT,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("tag", "--points-at", "HEAD"): "v0.14.1",
            ("rev-parse", "HEAD^{tree}"): "a" * 40,
            ("remote", "get-url", "origin"): "https://github.com/kmedved/chimeraboost.git",
        }
        if mutation == "head" and command == ("rev-parse", "HEAD"):
            return "0" * 40
        if mutation == "dirty" and command[0] == "status":
            return " M chimeraboost/__init__.py"
        if mutation == "tag" and command[0] == "tag":
            return ""
        return values[command]

    fake_module = SimpleNamespace(
        __file__=str(outside if mutation == "module" else module_path),
        __version__="0.0.0" if mutation == "version" else "0.14.1",
    )
    monkeypatch.setattr(campaign, "_run_git", mutated_git)
    monkeypatch.setattr(campaign.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(campaign.importlib, "import_module", lambda _: fake_module)
    if mutation == "warmup":
        monkeypatch.setenv("CHIMERABOOST_WARMUP", "1")
    else:
        monkeypatch.delenv("CHIMERABOOST_WARMUP", raising=False)
    old_path = list(sys.path)
    try:
        with pytest.raises(RuntimeError, match=expected_error):
            campaign.activate_chimeraboost_checkout(checkout)
        assert sys.path == old_path
    finally:
        sys.path[:] = old_path


@requires_benchmark_deps
def test_dry_run_performs_audit_without_any_filesystem_write(tmp_path, monkeypatch):
    output_dir = tmp_path / "must-not-exist"
    calls = []
    chimera_source = {"git_head": campaign.CHIMERABOOST_TAG_COMMIT}
    source = {
        "git_head": "b" * 40,
        "chimeraboost": chimera_source,
    }

    def validate_state(path, *, resume):
        calls.append((Path(path), resume))
        assert not Path(path).exists()

    monkeypatch.setattr(campaign.hardened, "validate_output_state", validate_state)
    monkeypatch.setattr(
        campaign,
        "activate_chimeraboost_checkout",
        lambda _: chimera_source,
    )
    monkeypatch.setattr(campaign, "_load_model_classes", lambda: {})
    monkeypatch.setattr(campaign, "validate_official_defaults", lambda _: None)
    monkeypatch.setattr(campaign, "build_experiments", lambda **_: {})
    monkeypatch.setattr(campaign, "build_comparator_jobs", lambda *_: [object()] * 135)
    monkeypatch.setattr(
        campaign,
        "ordering_audit",
        lambda _: {"job_order_sha256": "a" * 64, "lane_position_counts": {}},
    )
    monkeypatch.setattr(
        campaign, "resolve_and_pin_child_cpu_allocation", lambda _: 18
    )
    provenance_calls = []

    def collect_source_provenance(*, output_dir, chimeraboost_path):
        provenance_calls.append((output_dir, Path(chimeraboost_path)))
        return source

    manifest_calls = []

    def build_run_manifest(*, output_dir, source, ordering, resolved_child_num_cpus):
        manifest_calls.append(
            (output_dir, source, ordering, resolved_child_num_cpus)
        )
        return {
            "source": source,
            "ordering_audit": ordering,
            "resolved_child_num_cpus": resolved_child_num_cpus,
            "protocol_sha256": campaign.protocol_sha256(),
            "job_order_sha256": campaign.job_order_sha256(),
            "runtime": {"python_version": "3.12.13"},
        }

    monkeypatch.setattr(
        campaign, "collect_source_provenance", collect_source_provenance
    )
    monkeypatch.setattr(campaign, "build_run_manifest", build_run_manifest)
    monkeypatch.setattr(
        campaign.hardened,
        "_atomic_write_json",
        lambda *_args, **_kwargs: pytest.fail("dry-run attempted a write"),
    )
    monkeypatch.setattr(
        campaign,
        "_run_campaign",
        lambda **_: pytest.fail("dry-run entered the measured runner"),
        raising=False,
    )

    import tabarena.contexts as contexts
    import tabarena.utils.config_utils as config_utils

    monkeypatch.setattr(contexts, "TabArenaContext", lambda: object())
    monkeypatch.setattr(config_utils, "ConfigGenerator", object)
    assert (
        campaign.main(
            [
                "--output-dir",
                str(output_dir),
                "--chimeraboost-path",
                str(tmp_path / "chimera"),
                "--time-limit",
                "3600",
                "--dry-run",
            ]
        )
        == 0
    )
    assert calls == [(output_dir.resolve(), False)]
    assert provenance_calls == [
        (output_dir.resolve(), tmp_path / "chimera")
    ]
    assert manifest_calls == [
        (
            output_dir.resolve(),
            source,
            {"job_order_sha256": "a" * 64, "lane_position_counts": {}},
            18,
        )
    ]
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_output_state_rejects_symlinked_ancestor_for_fresh_and_resume(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output = linked_parent / "campaign"

    with pytest.raises(RuntimeError, match="symbolic-link component"):
        campaign._validate_comparator_output_state(output, resume=False)

    output.mkdir()
    with pytest.raises(RuntimeError, match="symbolic-link component"):
        campaign._validate_comparator_output_state(output, resume=True)


@requires_benchmark_deps
def test_campaign_dispatches_primary_then_diagnostic_as_two_task_contiguous_calls(
    tmp_path, monkeypatch
):
    jobs = campaign.order_comparator_jobs(_fake_jobs())
    ordering = campaign.ordering_audit(jobs)
    chimera_source = {"git_head": campaign.CHIMERABOOST_TAG_COMMIT}
    source = {"chimeraboost": chimera_source}
    manifest = {
        "source": source,
        "runtime": {},
        "ordering_audit": ordering,
    }
    warmup_calls = []
    dispatch_calls = []
    completion_calls = []

    class FakeContext:
        def run_jobs(self, lane_jobs, **kwargs):
            lane_jobs = list(lane_jobs)
            dispatch_calls.append((lane_jobs, kwargs))
            return [{"completed": index} for index in range(len(lane_jobs))]

    from benchmarks import tabarena_comparator_warmup as warmup_module

    monkeypatch.setattr(
        campaign,
        "collect_source_provenance",
        lambda **_: source,
    )
    monkeypatch.setattr(campaign, "build_run_manifest", lambda **_: manifest)
    monkeypatch.setattr(
        campaign, "write_or_validate_run_manifest", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(campaign, "prepare_grouped_resume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        warmup_module,
        "warmup_tabarena_comparators",
        lambda *, thread_count: warmup_calls.append(thread_count) or {"ok": True},
    )
    monkeypatch.setattr(campaign.hardened, "record_warmup", lambda *_: None)
    monkeypatch.setattr(
        campaign,
        "write_completion_attestation",
        lambda *args, **kwargs: completion_calls.append((args, kwargs)),
    )

    args = SimpleNamespace(
        chimeraboost_path=tmp_path / "chimera",
        resume=False,
    )
    assert campaign._run_campaign(
        args=args,
        output_dir=tmp_path / "campaign",
        context=FakeContext(),
        jobs=jobs,
        ordering=ordering,
        child_cpus=18,
        chimera_source=chimera_source,
    ) == 0

    primary, diagnostic = campaign.partition_jobs_for_dispatch(jobs)
    assert [call[0] for call in dispatch_calls] == [primary, diagnostic]
    assert [len(call[0]) for call in dispatch_calls] == [117, 18]
    assert all(
        call[1]
        == {
            "expname": str(tmp_path / "campaign" / "experiments"),
            "register": False,
            "new_result_prefix": "[same-machine D/M/C] ",
            "debug_mode": True,
        }
        for call in dispatch_calls
    )
    assert warmup_calls == [18]
    assert len(completion_calls) == 1
    assert completion_calls[0][1]["result_count"] == 135


def test_native_representation_validators_allow_only_attested_constant_drops():
    child_features = ["numeric", "constant", "category"]
    fitted_features = ["numeric", "category"]

    def digest(columns):
        return hashlib.sha256(
            json.dumps(columns, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    representation = {
        "schema_version": 2,
        "kind": "native",
        "fit_scope": "comparator_child_training_fold",
        "feature_alignment_policy": "autogluon_child_drop_unique",
        "target_used_by_representation": True,
        "input_feature_count": 3,
        "output_feature_count": 2,
        "external_feature_schema_sha256": digest(child_features),
        "fitted_feature_schema_sha256": digest(fitted_features),
        "categorical_input_columns": ["category"],
        "fitted_categorical_input_columns": ["category"],
        "dropped_constant_input_columns": ["constant"],
        "dropped_constant_input_unique_counts": [1],
    }
    arm = "darkofit_product_default"
    assert campaign._validate_representation(
        representation,
        arm=arm,
        dataset="airfoil_self_noise",
        field="native representation",
        child_features=child_features,
    ) == representation
    assert analysis._validate_representation(
        representation,
        lane="primary",
        dataset="airfoil_self_noise",
        child_features=child_features,
        field="native representation",
    ) == representation

    mutated = deepcopy(representation)
    mutated["dropped_constant_input_unique_counts"] = [2]
    with pytest.raises(RuntimeError, match="representation"):
        campaign._validate_representation(
            mutated,
            arm=arm,
            dataset="airfoil_self_noise",
            field="native representation",
            child_features=child_features,
        )
    with pytest.raises(RuntimeError, match="representation"):
        analysis._validate_representation(
            mutated,
            lane="primary",
            dataset="airfoil_self_noise",
            child_features=child_features,
            field="native representation",
        )


@requires_benchmark_deps
def test_safe_ordinal_validator_allows_partial_child_fold_observation():
    frame = _airfoil_frame()
    model = _instantiate(adapters.ComparatorOrdinalDarkoFitModel)
    model._fit_representation(frame)
    representation = model._representation_metadata()
    representation.update(
        {
            "fit_scope": "child_training_rows_only",
            "target_used_by_representation": False,
            "fit_calls": 1,
            "eval_transform_calls_during_fit": 1,
            "eval_unknown_counts": [0],
        }
    )
    assert representation["observed_training_category_counts"] == [3]
    child_features = list(frame.columns)
    assert campaign._validate_representation(
        representation,
        arm="darkofit_product_default_safe_ordinal",
        dataset="airfoil_self_noise",
        field="safe ordinal representation",
        child_features=child_features,
    ) == representation
    assert analysis._validate_representation(
        representation,
        lane="ordinal_diagnostic",
        dataset="airfoil_self_noise",
        child_features=child_features,
        field="safe ordinal representation",
    ) == representation


@requires_benchmark_deps
def test_complete_distribution_provenance_binds_catboost_and_autogluon_support():
    catboost = campaign._catboost_distribution_provenance()
    assert set(catboost["files"]) == {
        str(path)
        for path in campaign.metadata.distribution("catboost").files or ()
    }
    assert "catboost/metrics.py" in catboost["files"]
    analysis._verify_catboost_source(catboost)

    autogluon = campaign._installed_distribution_provenance(
        "autogluon.tabular",
        campaign.AUTOGLUON_PROVENANCE_MODULES["autogluon.tabular"],
    )
    assert set(autogluon["files"]) == {
        str(path)
        for path in campaign.metadata.distribution("autogluon.tabular").files or ()
    }
    assert "autogluon/tabular/models/catboost/callbacks.py" in autogluon["files"]
    assert (
        "autogluon/tabular/models/catboost/hyperparameters/parameters.py"
        in autogluon["files"]
    )
    analysis._verify_installed_distribution(
        autogluon,
        distribution_name="autogluon.tabular",
        module_names=campaign.AUTOGLUON_PROVENANCE_MODULES[
            "autogluon.tabular"
        ],
    )

    mutated = deepcopy(autogluon)
    mutated["files"][
        "autogluon/tabular/models/catboost/callbacks.py"
    ]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="installed bytes changed"):
        analysis._verify_installed_distribution(
            mutated,
            distribution_name="autogluon.tabular",
            module_names=campaign.AUTOGLUON_PROVENANCE_MODULES[
                "autogluon.tabular"
            ],
        )


@requires_benchmark_deps
def test_distribution_provenance_rejects_support_module_shadow(
    tmp_path, monkeypatch
):
    shadow = tmp_path / "callbacks.py"
    shadow.write_text("# shadow\n", encoding="utf-8")
    target = "autogluon.tabular.models.catboost.callbacks"
    real_import = campaign.importlib.import_module

    def import_module(name):
        if name == target:
            return SimpleNamespace(__file__=str(shadow))
        return real_import(name)

    monkeypatch.setattr(campaign.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="not in the attested distribution"):
        campaign._installed_distribution_provenance(
            "autogluon.tabular",
            campaign.AUTOGLUON_PROVENANCE_MODULES["autogluon.tabular"],
        )


@requires_benchmark_deps
def test_catboost_provenance_requires_imported_version(monkeypatch):
    real_import = campaign.importlib.import_module
    real_catboost = real_import("catboost")

    def import_module(name):
        if name == "catboost":
            return SimpleNamespace(__file__=real_catboost.__file__)
        return real_import(name)

    monkeypatch.setattr(campaign.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="version does not match"):
        campaign._catboost_distribution_provenance()


def test_distribution_provenance_rejects_record_hash_disagreement(
    tmp_path, monkeypatch
):
    installed = tmp_path / "fake.py"
    installed.write_bytes(b"real bytes")

    class FakePackagePath:
        hash = SimpleNamespace(mode="sha256", value="not-the-real-digest")
        size = len(b"real bytes")

        def __str__(self):
            return "fake.py"

    relative = FakePackagePath()
    distribution = SimpleNamespace(
        version="1.0",
        files=[relative],
        locate_file=lambda _relative: installed,
    )
    monkeypatch.setattr(
        campaign.metadata,
        "distribution",
        lambda _name: distribution,
    )
    with pytest.raises(RuntimeError, match="disagree with RECORD"):
        campaign._installed_distribution_provenance(
            "fake-distribution", ("fake.module",)
        )


@requires_benchmark_deps
def test_catboost_provenance_rejects_shadow_import(tmp_path, monkeypatch):
    shadow = tmp_path / "catboost.py"
    shadow.write_text("__version__ = '1.2.10'\n", encoding="utf-8")
    real_import = campaign.importlib.import_module

    def import_module(name):
        if name == "catboost":
            return SimpleNamespace(__file__=str(shadow), __version__="1.2.10")
        return real_import(name)

    monkeypatch.setattr(campaign.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="not in the attested distribution"):
        campaign._catboost_distribution_provenance()


def test_analyzer_treats_raw_results_as_opaque_bytes():
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    decoded_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            decoded_calls.append((None, node.func.id))
        elif isinstance(node.func, ast.Attribute):
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            decoded_calls.append((owner, node.func.attr))
    assert "pickle" not in imported
    assert ("pickle", "load") not in decoded_calls
    assert ("pickle", "loads") not in decoded_calls
    assert all(name not in {"read_pickle", "_decode_result_pickle"} for _, name in decoded_calls)
    assert "results.pkl" in source
    assert "read_bytes" in source or "_read_stable" in source


def test_seven_output_contract_is_deterministic_and_lane_specific():
    expected = (
        "primary_paired_splits.csv",
        "primary_per_dataset.csv",
        "ordinal_diagnostic_paired_splits.csv",
        "ordinal_diagnostic_per_dataset.csv",
        "paired_children.csv",
        "summary.json",
        "report.md",
    )
    assert campaign.DEFAULT_ANALYSIS_OUTPUT_FILENAMES == expected
    assert analysis.OUTPUT_NAMES == expected
    assert len(set(analysis.OUTPUT_KEYS)) == len(set(analysis.OUTPUT_NAMES)) == 7
    assert analysis.BOOTSTRAP_DRAWS == campaign.BOOTSTRAP_DRAWS == 10_000
    assert analysis.BOOTSTRAP_SEED == campaign.BOOTSTRAP_SEED == 20_260_719
    assert all("ordinal" not in name for name in expected[:2])
    assert all("ordinal_diagnostic" in name for name in expected[2:4])


def test_seven_output_targets_pass_the_generalized_safety_validator(tmp_path):
    protected = tmp_path / "manifest.json"
    protected.write_text("{}\n", encoding="utf-8")
    requested = {
        key: tmp_path / name
        for key, name in zip(analysis.OUTPUT_KEYS, analysis.OUTPUT_NAMES)
    }
    assert analysis._canonical_same_machine_output_targets(
        tmp_path,
        requested,
        protected_paths=[protected],
    ) == {key: path.resolve() for key, path in requested.items()}

    with pytest.raises(RuntimeError, match="fields are not exact"):
        analysis._canonical_same_machine_output_targets(
            tmp_path,
            dict(list(requested.items())[:-1]),
            protected_paths=[protected],
        )


def test_seven_output_publication_is_one_ordered_atomic_group(tmp_path, monkeypatch):
    outputs = {
        key: tmp_path / name
        for key, name in zip(analysis.OUTPUT_KEYS, analysis.OUTPUT_NAMES)
    }
    payloads = {key: f"new:{key}".encode() for key in analysis.OUTPUT_KEYS}
    observed = {}

    def capture_group(items, *, post_write_check):
        observed["items"] = list(items)
        observed["checked"] = False
        post_write_check()
        observed["checked"] = True

    monkeypatch.setattr(analysis.hardened, "_atomic_write_group", capture_group)
    analysis._publish_outputs_atomically(
        outputs, payloads, post_write_check=lambda: None
    )
    assert observed["items"] == [
        (outputs[key], payloads[key]) for key in analysis.OUTPUT_KEYS
    ]
    assert observed["checked"] is True

    missing = dict(payloads)
    missing.pop("report_md")
    with pytest.raises(RuntimeError, match="fields are not exact"):
        analysis._publish_outputs_atomically(
            outputs, missing, post_write_check=lambda: None
        )


def test_seven_output_publication_rolls_back_after_partial_install(
    tmp_path, monkeypatch
):
    outputs = {
        key: tmp_path / name
        for key, name in zip(analysis.OUTPUT_KEYS, analysis.OUTPUT_NAMES)
    }
    payloads = {key: f"new:{key}".encode() for key in analysis.OUTPUT_KEYS}
    for key, path in outputs.items():
        path.write_bytes(f"old:{key}".encode())

    original_replace = analysis.hardened.os.replace
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
            raise OSError("synthetic seven-output publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(analysis.hardened.os, "replace", fail_summary_install)
    with pytest.raises(OSError, match="synthetic seven-output"):
        analysis._publish_outputs_atomically(
            outputs, payloads, post_write_check=lambda: None
        )
    assert injected is True
    for key, path in outputs.items():
        assert path.read_bytes() == f"old:{key}".encode()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.backup")) == []


def test_analyzer_pairs_lanes_without_pooling_or_post_hoc_rescue():
    paired = analysis.pair_outer_rows(_minimal_outer_rows())
    assert len(paired) == 153
    assert sum(row["panel"] == "primary" for row in paired) == 117
    assert sum(row["panel"] == "ordinal_pairwise" for row in paired) == 18
    assert sum(row["panel"] == "ordinal_uplift" for row in paired) == 18
    assert {
        row["dataset"] for row in paired if row["panel"] != "primary"
    } == {"airfoil_self_noise", "diamonds"}
    assert all(
        row["numerator_lane"] == row["denominator_lane"] == "primary"
        for row in paired
        if row["panel"] == "primary"
    )
    assert all(
        row["numerator_lane"] == "ordinal_diagnostic"
        and row["denominator_lane"] == "primary"
        for row in paired
        if row["panel"] == "ordinal_uplift"
    )
    assert all(
        row[f"{metric}_ratio"] is None
        and row[f"numerator_{metric}"] is None
        and row[f"denominator_{metric}"] is None
        for row in paired
        if row["panel"] == "ordinal_uplift"
        for metric in set(analysis.METRICS) - set(analysis.ACCURACY_METRICS)
    )
    assert all(spec["panel"] != "policy_selection" for spec in analysis.contrast_specs())


def test_zero_incremental_memory_is_unavailable_without_a_pseudocount():
    unavailable = analysis._ratio_fields("incremental_memory_bytes", 0.0, 8192.0)
    assert unavailable == {
        "incremental_memory_bytes_ratio": None,
        "incremental_memory_bytes_log_ratio": None,
        "incremental_memory_bytes_pct": None,
        "incremental_memory_bytes_ratio_available": False,
        "incremental_memory_bytes_ratio_unavailable_reason": (
            "zero_incremental_memory_observation"
        ),
    }
    available = analysis._ratio_fields(
        "incremental_memory_bytes", 4096.0, 8192.0
    )
    assert available["incremental_memory_bytes_ratio"] == 0.5
    assert available["incremental_memory_bytes_log_ratio"] == pytest.approx(
        np.log(0.5)
    )
    assert available["incremental_memory_bytes_ratio_available"] is True


def test_equal_dataset_aggregation_and_bootstrap_are_exact_and_seeded():
    paired = [
        row
        for row in analysis.pair_outer_rows(_minimal_outer_rows())
        if row["panel"] == "primary" and row["contrast_code"] == "D/M"
    ]
    point, datasets = analysis.equal_dataset_point_log_ratio(
        paired, "test_rmse_log_ratio"
    )
    assert len(datasets) == 13
    assert point == pytest.approx(np.mean(list(datasets.values())), abs=1e-15)
    assert np.exp(point) == pytest.approx(0.5)

    first = analysis.fixed_dataset_bootstrap_log_ratios(
        paired, "test_rmse_log_ratio", draws=128, seed=20_260_719
    )
    second = analysis.fixed_dataset_bootstrap_log_ratios(
        paired, "test_rmse_log_ratio", draws=128, seed=20_260_719
    )
    assert np.array_equal(first, second)
    assert np.allclose(first, point)

    missing = paired[:-1]
    with pytest.raises(RuntimeError, match="coordinate set"):
        analysis.equal_dataset_point_log_ratio(missing, "test_rmse_log_ratio")
    duplicated = paired + [paired[0]]
    with pytest.raises(RuntimeError, match="duplicate"):
        analysis.fixed_dataset_bootstrap_log_ratios(
            duplicated, "test_rmse_log_ratio", draws=8
        )


def test_opaque_artifact_hash_and_size_attestation_rejects_byte_mutation(tmp_path):
    raw = tmp_path / "experiments" / "results.pkl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"opaque-not-decoded")
    metadata = {
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "size_bytes": raw.stat().st_size,
    }
    assert (
        analysis._artifact_bytes(
            tmp_path, "experiments/results.pkl", metadata, "raw result"
        )
        == b"opaque-not-decoded"
    )
    raw.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="attestation"):
        analysis._artifact_bytes(
            tmp_path, "experiments/results.pkl", metadata, "raw result"
        )


@requires_benchmark_deps
def test_completion_attestation_binds_all_counts_and_result_hashes(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / campaign.MANIFEST_FILENAME
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifacts = {
        "experiments/data/a/results.pkl": {
            "sha256": "a" * 64,
            "size_bytes": 123,
        }
    }
    source = {"git_head": "b" * 40, "chimeraboost": {"repository": "/chimera"}}
    runtime = {"python_version": "3.12.13"}
    manifest = {
        "protocol_sha256": campaign.protocol_sha256(),
        "job_order_sha256": campaign.job_order_sha256(),
        "ordering_audit": {"fixed": True},
        "source": source,
        "runtime": runtime,
    }
    monkeypatch.setattr(campaign, "ordering_audit", lambda _jobs: {"fixed": True})
    monkeypatch.setattr(
        campaign, "collect_result_artifacts", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        campaign,
        "validate_completed_results",
        lambda *_args, **_kwargs: (
            {"all_results_valid": True},
            [{"outer": True}],
            [{"child": True}],
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_history_artifact",
        lambda _root, filename, **_kwargs: (
            {"path": filename, "sha256": "c" * 64, "size_bytes": 1}
            if filename == campaign.WARMUP_HISTORY_FILENAME
            else None
        ),
    )
    monkeypatch.setattr(campaign, "collect_source_provenance", lambda **_: source)
    monkeypatch.setattr(campaign, "collect_runtime_provenance", lambda: runtime)
    attestation = campaign.write_completion_attestation(
        tmp_path,
        manifest=manifest,
        jobs=[object()] * 135,
        result_count=135,
    )
    assert attestation["result_count"] == 135
    assert attestation["expected_primary_result_count"] == 117
    assert attestation["expected_ordinal_diagnostic_result_count"] == 18
    assert attestation["expected_child_fits"] == 1_080
    assert attestation["result_artifacts"] == artifacts
    payload = json.loads(
        (tmp_path / campaign.ANALYSIS_PAYLOAD_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["result_artifacts_sha256"] == hashlib.sha256(
        campaign.hardened._canonical_json(artifacts)
    ).hexdigest()
    with pytest.raises(RuntimeError, match="expected 135"):
        campaign.write_completion_attestation(
            tmp_path, manifest=manifest, jobs=[], result_count=134
        )


def test_manifest_binds_protocol_order_source_runtime_and_exact_versions(tmp_path):
    source = {
        "repository": "/repo",
        "git_head": "a" * 40,
        "git_tree": "b" * 40,
        "files": {"runner.py": {"sha256": "c" * 64}},
        "chimeraboost": {
            "git_head": campaign.CHIMERABOOST_TAG_COMMIT,
            "git_tag": "v0.14.1",
        },
        "catboost": {"version": "1.2.10"},
    }
    runtime = {
        "python_executable": sys.executable,
        "python_version": "3.12.13",
        "packages": {"catboost": "1.2.10"},
        "environment": {"CHIMERABOOST_WARMUP": None},
        "hardware": {"logical_cpu_count": 18},
    }
    monkeypatch_runtime = pytest.MonkeyPatch()
    monkeypatch_runtime.setattr(campaign, "collect_runtime_provenance", lambda: runtime)
    try:
        ordering = {
            "job_order_sha256": campaign.job_order_sha256(),
            "lane_position_counts": {},
        }
        manifest = campaign.build_run_manifest(
            output_dir=tmp_path,
            source=source,
            ordering=ordering,
            resolved_child_num_cpus=18,
        )
    finally:
        monkeypatch_runtime.undo()
    assert manifest["kind"] == campaign.CAMPAIGN_KIND
    assert manifest["protocol_sha256"] == campaign.protocol_sha256()
    assert manifest["job_order_sha256"] == campaign.job_order_sha256()
    assert manifest["protocol"] == campaign.frozen_protocol()
    assert manifest["source"] == source
    assert manifest["runtime"] == runtime
    assert manifest["time_limit_seconds"] == 3_600.0
    assert manifest["resolved_child_num_cpus"] == 18
    mutated = deepcopy(manifest)
    mutated["protocol"]["expected_jobs"] = 134
    assert mutated["protocol"] != campaign.frozen_protocol()


def test_lane_specific_resume_invalidates_only_the_incomplete_triad(
    tmp_path, monkeypatch
):
    primary = ("primary", "airfoil_self_noise", 0, 0)
    diagnostic = ("ordinal_diagnostic", "airfoil_self_noise", 0, 0)
    coordinates = [primary, diagnostic]
    jobs = []
    for coordinate in coordinates:
        lane = coordinate[0]
        for arm, spec in campaign.ARM_SPECS.items():
            if spec["lane"] == lane:
                jobs.append(SimpleNamespace(arm=arm, coordinate=coordinate))

    def result_path(output_dir, job):
        return (
            output_dir
            / "experiments"
            / "data"
            / job.arm
            / "363612"
            / "0_0"
            / "results.pkl"
        )

    monkeypatch.setattr(
        campaign,
        "expected_coordinates",
        lambda lane=None: [row for row in coordinates if lane is None or row[0] == lane],
    )
    monkeypatch.setattr(campaign, "_job_arm", lambda job: job.arm)
    monkeypatch.setattr(campaign, "_job_coordinate", lambda job: job.coordinate)
    monkeypatch.setattr(campaign, "_result_path", result_path)
    monkeypatch.setattr(
        campaign,
        "_cached_result_issue",
        lambda path, _job: None if path.exists() else "missing",
    )

    missing_arm = campaign.ARM_BY_LANE_CODE[("ordinal_diagnostic", "C")]
    for job in jobs:
        path = result_path(tmp_path, job)
        if job.arm == missing_arm:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(job.arm.encode())

    record = campaign.prepare_grouped_resume(tmp_path, jobs, resume=True)
    assert record is not None
    assert record["invalidated_coordinate_count"] == 1
    assert record["invalidated_result_count"] == 2
    assert record["invalidated_coordinates"][0]["lane"] == "ordinal_diagnostic"
    assert record["invalidated_coordinates"][0]["arm_status"][missing_arm] == "missing"

    for job in jobs:
        path = result_path(tmp_path, job)
        if job.coordinate == primary:
            assert path.is_file()
        else:
            assert not path.exists()
    archived = list((tmp_path / "resume_invalidated").rglob("results.pkl"))
    assert len(archived) == 2


def test_failure_timeout_and_imputation_are_never_silently_admissible():
    source = "synthetic/results.pkl"
    for mutation, expected in (
        ({"problem_type": "classification", "metric": "rmse"}, "wrong problem"),
        ({"problem_type": "regression", "metric": "mae"}, "wrong problem"),
        (
            {"problem_type": "regression", "metric": "rmse", "imputed": True},
            "imputed",
        ),
    ):
        with pytest.raises(RuntimeError, match=expected):
            campaign.parse_result_record(mutation, source=source)

    for engine, arm in (
        ("darkofit", "darkofit_product_default"),
        ("chimeraboost", "chimeraboost_0_14_1_default"),
        ("catboost", "catboost_1_2_10_default"),
    ):
        fitted = _valid_comparator_fit(engine)
        fitted["stop_reason"] = "time_limit"
        if engine == "darkofit":
            fitted["deadline_hit"] = True
        with pytest.raises(RuntimeError, match="wall-clock"):
            campaign._validate_comparator_fit(
                fitted, arm=arm, field=f"{engine} timeout"
            )


def test_source_file_lists_bind_protocol_runner_analyzer_adapters_and_warmup():
    expected = {
        Path("benchmarks/tabarena_comparator_adapters.py"),
        Path("benchmarks/tabarena_comparator_warmup.py"),
        Path("benchmarks/run_tabarena_regression_same_machine.py"),
        Path("benchmarks/analyze_tabarena_regression_same_machine.py"),
        Path("benchmarks/tabarena_regression_same_machine_protocol.md"),
    }
    assert expected.issubset(set(campaign.SOURCE_FILES))
    assert "catboost" in campaign.PACKAGE_DISTRIBUTIONS
    assert "autogluon.features" in campaign.PACKAGE_DISTRIBUTIONS
    assert "graphviz" in campaign.PACKAGE_DISTRIBUTIONS
    assert "CHIMERABOOST_WARMUP" in campaign.RUNTIME_ENVIRONMENT_KEYS
    assert campaign.WARMUP_STAGE_NAMES == (
        "darkofit_numeric",
        "darkofit_categorical",
        "chimeraboost_numeric",
        "chimeraboost_categorical",
        "catboost_numeric",
        "catboost_categorical",
    )
