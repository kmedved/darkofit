from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks import run_smooth_group_linear_selector as experiment


def _result(task_id, config, rmse, selected=True):
    fit_metadata = {}
    if config == experiment.SELECTOR:
        fit_metadata = {
            "selected_linear_leaves": selected,
            "split": {"policy": "weighted_target_stratified"},
        }
    return {
        "task": {"task_id": task_id},
        "config": config,
        "folds": [
            {
                "fold": fold,
                "rmse": float(rmse),
                "fit_metadata": fit_metadata,
            }
            for fold in experiment.FOLDS
        ],
    }


def _results(selector=0.925, fixed=0.92):
    rows = []
    for task_number, task_id in enumerate(experiment.TASKS):
        values = {
            experiment.CONTROL: 1.0,
            experiment.SELECTOR: selector,
            experiment.FIXED: fixed,
            experiment.CHIMERA: 0.91,
        }
        for config, value in values.items():
            selected = not (
                config == experiment.SELECTOR and task_number == 0
            )
            rows.append(_result(task_id, config, value, selected))
    return rows


def test_analysis_advances_selective_smooth_policy():
    analysis = experiment.analyze(_results())

    assert analysis["passes_all_gates"] is True
    assert analysis["selection_count"] == 14
    assert analysis["decline_count"] == 7
    assert analysis["fixed_benefit_retention"] >= 0.9
    assert analysis["recommendation"] == (
        "advance_selector_to_fresh_confirmation_design"
    )


def test_analysis_closes_nonselective_or_regressing_policy():
    results = _results(selector=1.01)
    for row in results:
        if row["config"] == experiment.SELECTOR:
            for fold in row["folds"]:
                fold["fit_metadata"]["selected_linear_leaves"] = True

    analysis = experiment.analyze(results)

    assert analysis["passes_all_gates"] is False
    assert analysis["gates"]["equal_task_gain_at_least_2pct"] is False
    assert analysis["gates"]["declines_at_least_one_coordinate"] is False
    assert analysis["recommendation"] == "close_smooth_margin_selector"


def test_partition_boundary_keeps_lockbox_sealed():
    boundary = experiment.base._partition_boundary()
    assert not set(experiment.TASKS).intersection(
        boundary["lockbox_task_ids"]
    )


@pytest.mark.parametrize("stop_reason", ["early_stopping", "max_iterations"])
def test_selection_record_accepts_valid_early_stopping_outcomes(
    monkeypatch, stop_reason
):
    fitted = {"final_fit": {"stop_reason": stop_reason}}
    monkeypatch.setattr(
        experiment.basketball,
        "extract_fit_metadata",
        lambda model: fitted,
    )
    core = SimpleNamespace(
        auto_params_={"validation_split": {"source": "explicit_eval_set"}}
    )
    model = SimpleNamespace(model_=core, best_score_=1.0)

    record = experiment._selection_fit_record("constant", model, 1.0)

    assert record["fit_metadata"] == fitted
