from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_fresh_selector_confirmation as experiment


RECORDED_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "fresh_selector_confirmation.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d"
)


def _result(task_id, stratum, config, rmse, selected=False):
    return {
        "task_id": task_id,
        "dataset_name": f"task-{task_id}",
        "lineage_cluster": f"lineage-{task_id}",
        "stratum": stratum,
        "config": config,
        "folds": [
            {
                "rmse": float(rmse),
                "metadata": (
                    {"selected_linear_leaves": selected}
                    if config == experiment.SELECTOR
                    else {}
                ),
            }
            for _ in experiment.FOLDS
        ],
    }


def _results(primary_selector=0.95, chimera=0.96, noisy_selector=0.99):
    task_strata = {
        **{task_id: "smooth_process" for task_id in range(14)},
        **{task_id: "categorical" for task_id in range(14, 17)},
        **{task_id: "noisy_tabular" for task_id in range(17, 20)},
    }
    rows = []
    for task_id, stratum in task_strata.items():
        selector = (
            primary_selector
            if stratum == "smooth_process"
            else (noisy_selector if stratum == "noisy_tabular" else 0.99)
        )
        values = {
            experiment.CONTROL: 1.0,
            experiment.SELECTOR: selector,
            experiment.FIXED: 0.94,
            experiment.CHIMERA: chimera,
            experiment.CATBOOST: 0.93,
        }
        rows.extend(
            _result(
                task_id,
                stratum,
                config,
                value,
                selected=config == experiment.SELECTOR and task_id % 2 == 0,
            )
            for config, value in values.items()
        )
    return rows, task_strata


def test_analysis_advances_safe_selector_that_beats_chimera():
    results, strata = _results()
    analysis = experiment.analyze(results, strata)

    assert analysis["passes_all_gates"] is True
    assert analysis["selector_selection_count"] == 30
    assert analysis["recommendation"] == (
        "advance_to_lockbox_power_freeze"
    )


def test_analysis_closes_chimera_loss_or_noisy_regression():
    results, strata = _results(
        primary_selector=0.97,
        chimera=0.95,
        noisy_selector=1.01,
    )
    analysis = experiment.analyze(results, strata)

    assert analysis["passes_all_gates"] is False
    assert analysis["gates"]["primary_beats_chimeraboost_product"] is False
    assert analysis["gates"]["noisy_aggregate_nonregression"] is False
    assert analysis["recommendation"] == (
        "close_fresh_smooth_margin_selector"
    )


def test_registry_bindings_are_current():
    v1, v2, by_task, amendments = experiment._registries()

    assert v1["task_count"] == v2["task_count"] == 20
    assert len(by_task) == len(amendments) == 20
    assert sum(
        row["stratum"] == "smooth_process" for row in amendments.values()
    ) == 14


def test_split_identity_is_bound_to_frozen_indices():
    row = {
        "task_id": 7,
        "task_record": {
            "official_splits": {
                "coordinates": [
                    {
                        "repeat": 0,
                        "fold": 0,
                        "sample": 0,
                        "train_size": 3,
                        "test_size": 2,
                        "train_index_sha256": experiment._array_sha256(
                            [0, 2, 4], dtype="<i8"
                        ),
                        "test_index_sha256": experiment._array_sha256(
                            [1, 3], dtype="<i8"
                        ),
                    }
                ]
            }
        },
    }

    identity = experiment._verify_split(
        row,
        0,
        np.asarray([0, 2, 4]),
        np.asarray([1, 3]),
    )
    assert identity["train_size"] == 3

    with pytest.raises(RuntimeError, match="test_index_sha256 changed"):
        experiment._verify_split(
            row,
            0,
            np.asarray([0, 2, 4]),
            np.asarray([1, 4]),
        )


def test_catboost_report_only_best_iteration_can_be_absent():
    assert experiment._optional_int(None) is None
    assert experiment._optional_int(np.int64(7)) == 7


def test_recorded_artifact_closes_fresh_selector():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["sources"]["darkofit"]["clean"] is True
    assert artifact["sources"]["darkofit"]["head"] == (
        "29bd30cdcf476139c30efe4e09773ca812ba443f"
    )
    assert artifact["sources"]["chimeraboost"]["clean"] is True
    assert artifact["sources"]["chimeraboost"]["head"] == (
        experiment.EXPECTED_CHIMERA_HEAD
    )
    assert artifact["protocol"]["task_count"] == 20
    assert artifact["protocol"]["coordinate_count"] == 60
    assert artifact["protocol"]["lockbox_data_used"] is False
    assert artifact["protocol"]["task_drop_allowed"] is False
    assert artifact["protocol"]["task_imputation_allowed"] is False
    assert len(artifact["results"]) == 100
    analysis = artifact["analysis"]
    assert analysis["passes_all_gates"] is False
    assert analysis["selector_selection_count"] == 11
    assert analysis["selector_decline_count"] == 49
    assert analysis["gates"]["primary_ratio_at_most_0_98"] is False
    assert analysis["gates"]["primary_at_least_9_lineage_wins"] is False
    assert analysis["gates"]["primary_beats_chimeraboost_product"] is False
    assert analysis["recommendation"] == (
        "close_fresh_smooth_margin_selector"
    )
