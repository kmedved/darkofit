#!/usr/bin/env python3
"""Analyze the prospective T7b CatBoost-gap attribution campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks import analyze_t7_catboost_attribution as t7_analyzer
from benchmarks import run_t7b_catboost_gap_attribution as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = runner.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "t7b_catboost_gap_attribution_summary.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "benchmarks" / "t7b_catboost_gap_attribution_result.md"
)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 7_017
FAMILYWISE_ALPHA = 0.05
DIRECTIONAL_HYPOTHESES = 2 * (len(runner.ARM_NAMES) - 1)
PER_DIRECTION_ALPHA = FAMILYWISE_ALPHA / DIRECTIONAL_HYPOTHESES
LOWER_QUANTILE = PER_DIRECTION_ALPHA
UPPER_QUANTILE = 1.0 - PER_DIRECTION_ALPHA


def _finite_positive(value):
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _valid_score(score):
    return (
        isinstance(score, dict)
        and set(score) == {"rows", "rmse", "prediction_sha256"}
        and type(score["rows"]) is int
        and score["rows"] > 0
        and _finite_positive(score["rmse"])
        and type(score["prediction_sha256"]) is str
        and len(score["prediction_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in score["prediction_sha256"]
        )
    )


def _valid_source(source, protocol):
    return (
        isinstance(protocol, dict)
        and t7_analyzer._valid_source(source)
        and source["branch"] == "main"
        and source["clean"] is True
        and source["status"] == []
        and source["head"] == source["remote_branch_head"]
        and source["head"] == protocol.get("source_head")
    )


def _valid_runtime(runtime, freeze):
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "machine",
        "dependencies",
        "environment",
    }:
        return False
    descriptive = {
        key: runtime[key]
        for key in ("python", "machine", "dependencies")
    }
    return (
        t7_analyzer._valid_runtime(descriptive)
        and runtime["dependencies"]["catboost"] == "1.2.10"
        and runner.t7._same_typed_value(
            runtime["environment"], freeze["environment"]
        )
    )


def _expected_protocol(freeze, declaration, raw_protocol):
    try:
        runner._validate_execution_source(
            {"head": raw_protocol.get("source_head")}, freeze
        )
    except (AttributeError, RuntimeError, TypeError):
        return False
    return (
        isinstance(raw_protocol, dict)
        and set(raw_protocol)
        == {
            "freeze_file_sha256",
            "freeze_canonical_sha256",
            "source_sha256",
            "coordinates_sha256",
            "source_head",
            "model_source_head",
            "environment",
            "catboost_version",
            "task_type",
            "seeds",
            "arms",
            "historical_darkofit_over_catboost_default_ratio",
        }
        and raw_protocol["freeze_file_sha256"] == runner._sha256(runner.FREEZE)
        and raw_protocol["freeze_canonical_sha256"] == freeze["freeze_sha256"]
        and runner.t7._same_typed_value(
            raw_protocol["source_sha256"], freeze["source_sha256"]
        )
        and raw_protocol["coordinates_sha256"]
        == declaration["coordinates_sha256"]
        and type(raw_protocol["source_head"]) is str
        and len(raw_protocol["source_head"]) == 40
        and all(
            character in "0123456789abcdef"
            for character in raw_protocol["source_head"]
        )
        and raw_protocol["model_source_head"]
        == freeze["darkofit_model_head"]
        and runner.t7._same_typed_value(
            raw_protocol["environment"], freeze["environment"]
        )
        and raw_protocol["catboost_version"] == "1.2.10"
        and raw_protocol["task_type"] == "CPU"
        and raw_protocol["seeds"] == list(runner.SEEDS)
        and raw_protocol["arms"] == list(runner.ARM_NAMES)
        and raw_protocol[
            "historical_darkofit_over_catboost_default_ratio"
        ]
        == declaration["input_boundary"][
            "historical_darkofit_over_catboost_default_ratio"
        ]
    )


def _validate_arm(
    arm, expected_name, coordinate, seed, categorical_count
):
    expected_keys = {
        "arm",
        "position",
        "overrides",
        "fit_seconds",
        "validation",
        "test",
        "prediction_timing",
        "tree_count",
        "requested_policy",
        "constructor_params_observed",
        "resolved_params",
    }
    if (
        not isinstance(arm, dict)
        or set(arm) != expected_keys
        or arm["arm"] != expected_name
        or type(arm["position"]) is not int
        or not runner.t7._same_typed_value(
            arm["overrides"], runner.ARMS[expected_name]
        )
        or not _finite_positive(arm["fit_seconds"])
        or not _valid_score(arm["validation"])
        or not _valid_score(arm["test"])
        or type(arm["tree_count"]) is not int
        or arm["tree_count"] != runner.ITERATIONS
        or not isinstance(arm["requested_policy"], dict)
        or not isinstance(arm["constructor_params_observed"], dict)
        or not isinstance(arm["resolved_params"], dict)
        or set(arm["resolved_params"]) != set(runner.RESOLVED_KEYS)
    ):
        raise RuntimeError("invalid T7b arm record")
    timing = arm["prediction_timing"]
    if (
        not isinstance(timing, dict)
        or set(timing) != {"calls", "median_seconds", "total_seconds"}
        or type(timing["calls"]) is not int
        or timing["calls"] != runner.t7.PREDICTION_CALLS
        or not _finite_positive(timing["median_seconds"])
        or not _finite_positive(timing["total_seconds"])
    ):
        raise RuntimeError("invalid T7b timing record")
    runner._validate_resolved(
        arm["resolved_params"],
        coordinate,
        seed,
        expected_name,
        categorical_count=categorical_count,
    )
    runner._validate_requested(
        arm["requested_policy"], coordinate, seed, expected_name
    )
    runner._validate_constructor_params_observed(
        arm["constructor_params_observed"]
    )


def _validate_result(result, expected, coordinate, frozen):
    expected_keys = {
        "execution_index",
        "coordinate_index",
        "task_id",
        "dataset_id",
        "dataset_name",
        "lineage_cluster",
        "fold",
        "seed",
        "frozen_learning_rate",
        "n_features",
        "categorical_count",
        "outer_split",
        "inner_split",
        "arm_order",
        "arms",
        "warmup_seconds",
        "peak_rss_bytes",
        "integrity",
        "behavior_sha256",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected_keys
        or any(
            type(result[key]) is not int
            for key in (
                "execution_index",
                "coordinate_index",
                "task_id",
                "dataset_id",
                "fold",
                "seed",
                "n_features",
                "categorical_count",
                "peak_rss_bytes",
            )
        )
        or result["execution_index"] != expected["execution_index"]
        or result["coordinate_index"] != expected["coordinate_index"]
        or result["task_id"] != expected["task_id"]
        or result["dataset_id"] != frozen["dataset_id"]
        or result["dataset_name"] != frozen["dataset_name"]
        or result["lineage_cluster"] != frozen["lineage_cluster"]
        or result["fold"] != expected["fold"]
        or result["seed"] != expected["seed"]
        or type(result["dataset_name"]) is not str
        or type(result["lineage_cluster"]) is not str
        or type(result["frozen_learning_rate"]) is not float
        or result["frozen_learning_rate"]
        != coordinate["frozen_learning_rate"]
        or result["n_features"] != frozen["n_features"]
        or result["categorical_count"] != frozen["categorical_count"]
        or not runner.t7._same_typed_value(
            result["outer_split"], frozen["outer_split"]
        )
        or not runner.t7._same_typed_value(
            result["inner_split"], frozen["inner_split"]
        )
        or result["arm_order"]
        != list(runner._arm_order(expected["execution_index"]))
        or not isinstance(result["arms"], list)
        or len(result["arms"]) != len(runner.ARM_NAMES)
        or not _finite_positive(result["warmup_seconds"])
        or result["peak_rss_bytes"] <= 0
        or type(result["behavior_sha256"]) is not str
        or len(result["behavior_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in result["behavior_sha256"]
        )
    ):
        raise RuntimeError("invalid T7b result record")
    for position, (arm, name) in enumerate(
        zip(result["arms"], result["arm_order"])
    ):
        _validate_arm(
            arm,
            name,
            coordinate,
            result["seed"],
            result["categorical_count"],
        )
        if arm["position"] != position:
            raise RuntimeError("invalid T7b arm position")
    behavior_sha256 = result["behavior_sha256"]
    expected_behavior = runner._json_sha256(
        {
            "execution_index": result["execution_index"],
            "task_id": result["task_id"],
            "fold": result["fold"],
            "seed": result["seed"],
            "arms": [
                {
                    "arm": arm["arm"],
                    "validation": arm["validation"],
                    "test": arm["test"],
                    "requested_policy": arm["requested_policy"],
                    "constructor_params_observed": arm[
                        "constructor_params_observed"
                    ],
                    "resolved_params": arm["resolved_params"],
                }
                for arm in result["arms"]
            ],
            "integrity": result["integrity"],
        }
    )
    if behavior_sha256 != expected_behavior:
        raise RuntimeError("T7b behavior hash changed")
    integrity_copy = dict(result)
    runner._integrity_checks(integrity_copy, coordinate)
    if not runner.t7._same_typed_value(
        integrity_copy["integrity"], result["integrity"]
    ):
        raise RuntimeError("T7b integrity record changed")


def load(path=DEFAULT_INPUT, *, return_file_sha256=False):
    freeze = runner._source_freeze(verify_live=False)
    declaration, frozen_by_coordinate = runner._coordinates(
        source_head=freeze["darkofit_model_head"]
    )
    encoded = Path(path).read_bytes()
    raw_file_sha256 = hashlib.sha256(encoded).hexdigest()
    raw = runner.t7._json_loads(encoded, "T7b raw artifact")
    expected_keys = {
        "schema_version",
        "name",
        "created_at",
        "development_data_only",
        "lockbox_data_used",
        "confirmation_outcomes_inspected",
        "default_change_authorized",
        "source",
        "runtime",
        "protocol",
        "coordinate_count",
        "execution_count",
        "fit_count",
        "resumed_execution_count",
        "results",
        "spool_records",
        "raw_sha256",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_keys
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["name"] != "darkofit_t7b_catboost_gap_attribution_raw_v1"
        or type(raw["created_at"]) is not str
        or raw["development_data_only"] is not True
        or raw["lockbox_data_used"] is not False
        or raw["confirmation_outcomes_inspected"] is not False
        or raw["default_change_authorized"] is not False
        or not _valid_source(raw["source"], raw["protocol"])
        or not _valid_runtime(raw["runtime"], freeze)
        or type(raw["coordinate_count"]) is not int
        or raw["coordinate_count"] != 24
        or type(raw["execution_count"]) is not int
        or raw["execution_count"] != 72
        or type(raw["fit_count"]) is not int
        or raw["fit_count"] != 576
        or type(raw["resumed_execution_count"]) is not int
        or not 0 <= raw["resumed_execution_count"] <= 72
        or not _expected_protocol(freeze, declaration, raw["protocol"])
        or not isinstance(raw["results"], list)
        or len(raw["results"]) != 72
        or not isinstance(raw["spool_records"], list)
        or len(raw["spool_records"]) != 72
    ):
        raise RuntimeError("invalid T7b raw artifact")
    try:
        created_at = datetime.fromisoformat(raw["created_at"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid T7b raw creation timestamp") from error
    if created_at.tzinfo is None:
        raise RuntimeError("invalid T7b raw creation timestamp")
    canonical = dict(raw)
    raw_sha256 = canonical.pop("raw_sha256")
    if type(raw_sha256) is not str or raw_sha256 != runner._json_sha256(canonical):
        raise RuntimeError("T7b raw hash changed")
    schedule = runner._schedule(declaration["coordinates"])
    if [result.get("execution_index") for result in raw["results"]] != list(
        range(72)
    ):
        raise RuntimeError("T7b result order changed")
    for result, expected in zip(raw["results"], schedule):
        coordinate = declaration["coordinates"][expected["coordinate_index"]]
        frozen = frozen_by_coordinate[expected["coordinate_index"]]
        _validate_result(result, expected, coordinate, frozen)
    expected_spool = {}
    for result, expected in zip(raw["results"], schedule):
        name = runner._spool_path(
            Path("/spool"),
            expected["task_id"],
            expected["fold"],
            expected["seed"],
        ).name
        expected_spool[name] = _spool_digest(raw["protocol"], result)
    observed_spool = {}
    for record in raw["spool_records"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "resumed"}
            or type(record["path"]) is not str
            or Path(record["path"]).name != record["path"]
            or type(record["sha256"]) is not str
            or len(record["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record["sha256"]
            )
            or type(record["resumed"]) is not bool
        ):
            raise RuntimeError("invalid T7b spool record")
        name = Path(record["path"]).name
        if name in observed_spool:
            raise RuntimeError("duplicate T7b spool record")
        observed_spool[name] = record["sha256"]
    if observed_spool != expected_spool:
        raise RuntimeError("T7b spool coverage changed")
    if sum(record["resumed"] for record in raw["spool_records"]) != raw[
        "resumed_execution_count"
    ]:
        raise RuntimeError("T7b resumed spool count changed")
    return (raw, raw_file_sha256) if return_file_sha256 else raw


def _spool_digest(binding, result):
    payload = {
        "binding": binding,
        "task_id": result["task_id"],
        "fold": result["fold"],
        "result_sha256": runner._json_sha256(result),
        "result": result,
    }
    return runner._json_sha256(payload)


def _geomean(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _ratio_records(results, arm_name, metric):
    records = {}
    for result in results:
        by_arm = {arm["arm"]: arm for arm in result["arms"]}
        ratio = (
            float(by_arm[arm_name][metric]["rmse"])
            / float(by_arm["baseline"][metric]["rmse"])
        )
        records.setdefault(result["task_id"], []).append(
            {
                "fold": result["fold"],
                "seed": result["seed"],
                "ratio": ratio,
                "dataset_name": result["dataset_name"],
            }
        )
    if len(records) != 8 or any(len(rows) != 9 for rows in records.values()):
        raise RuntimeError("T7b contrast coverage changed")
    return records


def _hierarchical_bounds(records, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if type(draws) is not int or draws <= 0 or type(seed) is not int:
        raise ValueError("invalid T7b bootstrap settings")
    task_ids = sorted(records)
    logs = {}
    for task_id in task_ids:
        by_fold = {}
        for row in records[task_id]:
            by_fold.setdefault(row["fold"], {})[row["seed"]] = math.log(
                float(row["ratio"])
            )
        if (
            set(by_fold) != set(runner.FOLDS)
            or any(set(values) != set(runner.SEEDS) for values in by_fold.values())
        ):
            raise ValueError("T7b bootstrap coordinate coverage changed")
        logs[task_id] = by_fold
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected_tasks = rng.integers(
            0, len(task_ids), size=len(task_ids)
        )
        task_means = []
        for task_index in selected_tasks:
            task = logs[task_ids[int(task_index)]]
            selected_folds = rng.integers(
                0, len(runner.FOLDS), size=len(runner.FOLDS)
            )
            values = []
            for fold_index in selected_folds:
                fold = runner.FOLDS[int(fold_index)]
                values.extend(
                    task[fold][seed]
                    for seed in runner.SEEDS
                )
            task_means.append(float(np.mean(values)))
        samples[draw] = math.exp(float(np.mean(task_means)))
    return {
        "draws": draws,
        "seed": seed,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "directional_hypotheses": DIRECTIONAL_HYPOTHESES,
        "per_direction_alpha": PER_DIRECTION_ALPHA,
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "bonferroni_lower": float(np.quantile(samples, LOWER_QUANTILE)),
        "bonferroni_upper": float(np.quantile(samples, UPPER_QUANTILE)),
        "seed_treatment": "fixed_average_within_sampled_fold",
    }


def _contrast(results, arm_name, historical_gap):
    test_records = _ratio_records(results, arm_name, "test")
    validation_records = _ratio_records(results, arm_name, "validation")
    task_ids = sorted(test_records)
    per_task = []
    for task_id in task_ids:
        test_ratio = _geomean(
            row["ratio"] for row in test_records[task_id]
        )
        validation_ratio = _geomean(
            row["ratio"] for row in validation_records[task_id]
        )
        per_task.append(
            {
                "task_id": task_id,
                "dataset_name": test_records[task_id][0]["dataset_name"],
                "test_ratio": test_ratio,
                "validation_ratio": validation_ratio,
            }
        )
    equal_test = _geomean(row["test_ratio"] for row in per_task)
    equal_validation = _geomean(
        row["validation_ratio"] for row in per_task
    )
    seed_ratios = {}
    for seed in runner.SEEDS:
        seed_ratios[str(seed)] = _geomean(
            _geomean(
                row["ratio"]
                for row in test_records[task_id]
                if row["seed"] == seed
            )
            for task_id in task_ids
        )
    leave_one_out = {
        str(task_id): _geomean(
            row["test_ratio"]
            for row in per_task
            if row["task_id"] != task_id
        )
        for task_id in task_ids
    }
    bootstrap = _hierarchical_bounds(test_records)
    minimum_loo = min(leave_one_out.values())
    maximum_loo = max(leave_one_out.values())
    worst_task = max(row["test_ratio"] for row in per_task)
    contributor_gates = {
        "equal_dataset_test_ratio_gt_1": equal_test > 1,
        "equal_dataset_validation_ratio_gt_1": equal_validation > 1,
        "bonferroni_lower_gt_1": bootstrap["bonferroni_lower"] > 1,
        "every_leave_one_task_out_ratio_gt_1": minimum_loo > 1,
        "every_seed_block_ratio_gt_1": min(seed_ratios.values()) > 1,
    }
    promising_gates = {
        "equal_dataset_test_ratio_lte_0_995": equal_test <= 0.995,
        "equal_dataset_validation_ratio_lte_1_005": (
            equal_validation <= 1.005
        ),
        "bonferroni_upper_lt_1": bootstrap["bonferroni_upper"] < 1,
        "worst_task_test_ratio_lte_1_02": worst_task <= 1.02,
        "every_leave_one_task_out_ratio_lte_1": maximum_loo <= 1,
        "every_seed_block_ratio_lte_1_005": (
            max(seed_ratios.values()) <= 1.005
        ),
    }
    contributor = all(contributor_gates.values())
    promising = all(promising_gates.values())
    label = (
        "contributor"
        if contributor
        else "promising_config"
        if promising
        else "not_attributed"
    )
    tolerance = 1e-12
    task_log_ratios = [math.log(row["test_ratio"]) for row in per_task]
    return {
        "arm": arm_name,
        "label": label,
        "equal_dataset_test_ratio": equal_test,
        "equal_dataset_validation_ratio": equal_validation,
        "historical_gap_fraction_erased_seed4_bridge": (
            math.log(seed_ratios["4"]) / math.log(historical_gap)
        ),
        "historical_gap_fraction_erased_three_seed_sensitivity": (
            math.log(equal_test) / math.log(historical_gap)
        ),
        "task_wins": sum(
            value < -tolerance for value in task_log_ratios
        ),
        "task_losses": sum(
            value > tolerance for value in task_log_ratios
        ),
        "task_ties": sum(
            abs(value) <= tolerance for value in task_log_ratios
        ),
        "worst_task_test_ratio": worst_task,
        "least_favorable_contributor_loo_ratio": minimum_loo,
        "least_favorable_promising_loo_ratio": maximum_loo,
        "seed_test_ratios": seed_ratios,
        "hierarchical_bootstrap": bootstrap,
        "contributor_gates": contributor_gates,
        "promising_config_gates": promising_gates,
        "per_task": per_task,
    }


def _descriptive_runtime(results):
    by_arm = {arm: {"fit": [], "predict": []} for arm in runner.ARM_NAMES}
    peaks = []
    warmups = []
    for result in results:
        peaks.append(result["peak_rss_bytes"])
        warmups.append(result["warmup_seconds"])
        for arm in result["arms"]:
            by_arm[arm["arm"]]["fit"].append(arm["fit_seconds"])
            by_arm[arm["arm"]]["predict"].append(
                arm["prediction_timing"]["median_seconds"]
            )
    baseline_fit = float(np.median(by_arm["baseline"]["fit"]))
    baseline_predict = float(np.median(by_arm["baseline"]["predict"]))
    return {
        "timing_is_descriptive_only": True,
        "per_arm": {
            arm: {
                "fit_median_seconds": float(np.median(values["fit"])),
                "fit_geomean_seconds": _geomean(values["fit"]),
                "fit_median_over_baseline": (
                    float(np.median(values["fit"])) / baseline_fit
                ),
                "prediction_median_seconds": float(
                    np.median(values["predict"])
                ),
                "prediction_geomean_seconds": _geomean(values["predict"]),
                "prediction_median_over_baseline": (
                    float(np.median(values["predict"])) / baseline_predict
                ),
            }
            for arm, values in by_arm.items()
        },
        "execution_peak_rss_bytes": {
            "median": float(np.median(peaks)),
            "maximum": int(max(peaks)),
        },
        "warmup_seconds": {
            "median": float(np.median(warmups)),
            "maximum": float(max(warmups)),
        },
    }


def _incremental_vs_components(by_arm):
    combined = by_arm["no_split_noise_or_row_sampling"]
    components = ("random_strength_0", "bootstrap_no")
    direct = {}
    for component in components:
        row = by_arm[component]
        component_tasks = {
            task["task_id"]: task for task in row["per_task"]
        }
        direct[component] = {
            "combined_over_component_test_ratio": (
                combined["equal_dataset_test_ratio"]
                / row["equal_dataset_test_ratio"]
            ),
            "combined_over_component_validation_ratio": (
                combined["equal_dataset_validation_ratio"]
                / row["equal_dataset_validation_ratio"]
            ),
            "seed_test_ratios": {
                seed: combined["seed_test_ratios"][seed]
                / row["seed_test_ratios"][seed]
                for seed in map(str, runner.SEEDS)
            },
            "per_task_test_ratios": {
                str(task["task_id"]): (
                    task["test_ratio"]
                    / component_tasks[task["task_id"]]["test_ratio"]
                )
                for task in combined["per_task"]
            },
        }
    direct_test = [
        direct[component]["combined_over_component_test_ratio"]
        for component in components
    ]
    if combined["label"] == "contributor":
        status = (
            "explanatory_incremental_vs_both_components_descriptive"
            if min(direct_test) >= 1.005
            else "explanatory_not_incremental_vs_both_components_descriptive"
        )
    elif combined["label"] == "promising_config":
        status = (
            "promising_incremental_vs_both_components_descriptive"
            if max(direct_test) <= 0.995
            else "promising_not_incremental_vs_both_components_descriptive"
        )
    else:
        status = "not_supported"
    return {
        "status": status,
        "inferential_claim_authorized": False,
        "incremental_ratio_threshold": 0.005,
        "direct_contrasts": direct,
        "paired_additive_log_departure": (
            math.log(combined["equal_dataset_test_ratio"])
            - math.log(by_arm["random_strength_0"]["equal_dataset_test_ratio"])
            - math.log(by_arm["bootstrap_no"]["equal_dataset_test_ratio"])
        ),
    }


def analyze(raw, *, raw_file_sha256):
    if (
        type(raw_file_sha256) is not str
        or len(raw_file_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in raw_file_sha256
        )
    ):
        raise ValueError("T7b analysis requires the exact raw file SHA-256")
    historical_gap = raw["protocol"][
        "historical_darkofit_over_catboost_default_ratio"
    ]
    contrasts = [
        _contrast(raw["results"], arm, historical_gap)
        for arm in runner.ARM_NAMES
        if arm != "baseline"
    ]
    by_arm = {row["arm"]: row for row in contrasts}
    summary = {
        "schema_version": 1,
        "name": "darkofit_t7b_catboost_gap_attribution_summary_v1",
        "development_data_only": True,
        "lockbox_data_used": False,
        "default_change_authorized": False,
        "decision": "development_attribution_only",
        "raw_file_sha256": raw_file_sha256,
        "raw_canonical_sha256": raw["raw_sha256"],
        "source": raw["source"],
        "runtime": raw["runtime"],
        "historical_darkofit_over_catboost_default_ratio": historical_gap,
        "multiplicity_control": {
            "familywise_alpha": FAMILYWISE_ALPHA,
            "directional_hypotheses": DIRECTIONAL_HYPOTHESES,
            "per_direction_alpha": PER_DIRECTION_ALPHA,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "contrasts": contrasts,
        "contributors": [
            row["arm"] for row in contrasts if row["label"] == "contributor"
        ],
        "promising_configs": [
            row["arm"]
            for row in contrasts
            if row["label"] == "promising_config"
        ],
        "noise_sampling_incremental_vs_components": (
            _incremental_vs_components(by_arm)
        ),
        "descriptive_runtime": _descriptive_runtime(raw["results"]),
    }
    summary["summary_sha256"] = runner._json_sha256(summary)
    return summary


def render(summary):
    rows = "\n".join(
        "| {arm} | {test:.6f} | {validation:.6f} | {lower:.6f} | "
        "{upper:.6f} | {seed4:.1%} | {all_seed:.1%} | {worst:.6f} | "
        "{contributor_loo:.6f} | {promising_loo:.6f} | `{label}` |".format(
            arm=row["arm"],
            test=row["equal_dataset_test_ratio"],
            validation=row["equal_dataset_validation_ratio"],
            lower=row["hierarchical_bootstrap"]["bonferroni_lower"],
            upper=row["hierarchical_bootstrap"]["bonferroni_upper"],
            seed4=row["historical_gap_fraction_erased_seed4_bridge"],
            all_seed=(
                row[
                    "historical_gap_fraction_erased_three_seed_sensitivity"
                ]
            ),
            worst=row["worst_task_test_ratio"],
            contributor_loo=row[
                "least_favorable_contributor_loo_ratio"
            ],
            promising_loo=row[
                "least_favorable_promising_loo_ratio"
            ],
            label=row["label"],
        )
        for row in summary["contrasts"]
    )
    seed_rows = "\n".join(
        f"| {row['arm']} | "
        + " | ".join(
            f"{row['seed_test_ratios'][str(seed)]:.6f}"
            for seed in runner.SEEDS
        )
        + " |"
        for row in summary["contrasts"]
    )
    task_rows = "\n".join(
        "| {arm} | {task_id} | {name} | {test:.6f} | "
        "{validation:.6f} |".format(
            arm=row["arm"],
            task_id=task["task_id"],
            name=task["dataset_name"],
            test=task["test_ratio"],
            validation=task["validation_ratio"],
        )
        for row in summary["contrasts"]
        for task in row["per_task"]
    )
    gate_rows = "\n".join(
        f"| {row['arm']} | {name} | {'pass' if passed else 'fail'} |"
        for row in summary["contrasts"]
        for family in ("contributor_gates", "promising_config_gates")
        for name, passed in row[family].items()
    )
    timing_rows = "\n".join(
        "| {arm} | {fit:.6f} | {fit_ratio:.3f} | {predict:.6f} | "
        "{predict_ratio:.3f} |".format(
            arm=arm,
            fit=values["fit_median_seconds"],
            fit_ratio=values["fit_median_over_baseline"],
            predict=values["prediction_median_seconds"],
            predict_ratio=values["prediction_median_over_baseline"],
        )
        for arm, values in summary["descriptive_runtime"]["per_arm"].items()
    )
    incremental = summary["noise_sampling_incremental_vs_components"]
    incremental_rows = "\n".join(
        "| {component} | {test:.6f} | {validation:.6f} |".format(
            component=component,
            test=values["combined_over_component_test_ratio"],
            validation=values["combined_over_component_validation_ratio"],
        )
        for component, values in incremental["direct_contrasts"].items()
    )
    control = summary["multiplicity_control"]
    rss = summary["descriptive_runtime"]["execution_peak_rss_bytes"]
    return f"""# T7b CatBoost-gap attribution

**Decision: `{summary['decision']}`.** This is spent development evidence and
does not authorize a default change.

Historical DarkoFit / CatBoost default RMSE ratio:
`{summary['historical_darkofit_over_catboost_default_ratio']:.6f}`.

The seed-4 bridge reports the fraction of that historical numerical gap erased
by a CatBoost perturbation. It is not a causal fraction explained. The
three-seed value is a separate sensitivity estimate.

Multiplicity uses {control['bootstrap_draws']:,} deterministic bootstrap draws,
familywise alpha {control['familywise_alpha']:.3f}, and
{control['directional_hypotheses']} directional hypotheses. The Bonferroni
quantiles are {control['lower_quantile']:.6f} and
{control['upper_quantile']:.6f}. Seeds are fixed repeat blocks and are averaged,
not independently resampled within each fold.

| Arm | Test ratio | Validation ratio | Bonferroni lower | Bonferroni upper | Seed-4 gap erased | Three-seed sensitivity | Worst task | Contributor LOO | Promising LOO | Label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

## Seed blocks

| Arm | Seed 4 | Seed 17 | Seed 29 |
|---|---:|---:|---:|
{seed_rows}

## Gate evidence

| Arm | Gate | Outcome |
|---|---|---|
{gate_rows}

## Per-task ratios

| Arm | Task | Dataset | Test ratio | Validation ratio |
|---|---:|---|---:|---:|
{task_rows}

## Noise/sampling incremental comparison

This comparison is descriptive and does not authorize an interaction claim.
Status: `{incremental['status']}`. Paired additive log departure:
`{incremental['paired_additive_log_departure']:.6f}`.

| Combined over component | Test ratio | Validation ratio |
|---|---:|---:|
{incremental_rows}

## Descriptive runtime

These measurements were collected under concurrent execution and are not
inferential gates.

| Arm | Median fit seconds | Fit / baseline | Median predict seconds | Predict / baseline |
|---|---:|---:|---:|---:|
{timing_rows}

Execution peak RSS: median {rss['median']:.0f} bytes; maximum
{rss['maximum']} bytes.

Contributors: {summary['contributors'] or 'none'}.
Promising configurations: {summary['promising_configs'] or 'none'}.

The explanatory direction requires an ablation to worsen CatBoost; the
promising-configuration direction requires an improvement. No confirmation
or lockbox data was opened.

Raw file SHA-256: `{summary['raw_file_sha256']}`.
Raw canonical SHA-256: `{summary['raw_canonical_sha256']}`.
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    args.input = Path(os.path.abspath(os.path.expanduser(args.input)))
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.markdown = Path(os.path.abspath(os.path.expanduser(args.markdown)))
    return args


def main(argv=None):
    args = parse_args(argv)
    raw, raw_file_sha256 = load(
        args.input, return_file_sha256=True
    )
    summary = analyze(raw, raw_file_sha256=raw_file_sha256)
    encoded = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    markdown = render(summary).encode()
    t7_analyzer._atomic_create_pair(
        args.output, encoded, args.markdown, markdown
    )


if __name__ == "__main__":
    main()
