from __future__ import annotations

from benchmarks import run_smooth_linear_leaves_development as experiment


def _result(task_id, config, rmse):
    return {
        "task": {"task_id": task_id},
        "config": config,
        "folds": [
            {"fold": fold, "rmse": float(rmse)}
            for fold in experiment.FOLDS
        ],
    }


def _results(linear=0.95, matched=0.94, residual=0.99, chimera=0.90):
    rows = []
    for task_id in experiment.TASKS:
        values = {
            "darko_default": 1.0,
            "darko_linear_current": linear,
            "darko_linear_matched": matched,
            "darko_linear_residual": residual,
            "chimera_linear_only": 0.96,
            "chimera_product": chimera,
        }
        rows.extend(
            _result(task_id, config, value)
            for config, value in values.items()
        )
    return rows


def test_analysis_advances_best_linear_and_deprecates_residual():
    analysis = experiment.analyze(_results())

    assert analysis["advancing_candidate"] == "darko_linear_matched"
    assert analysis["advances_to_selector_design"]
    assert analysis["recommend_linear_residual_deprecation"]
    assert not analysis["chimera_product_parity_reached"]
    assert analysis["recommendation"] == (
        "design_selector_and_deprecate_linear_residual"
    )


def test_analysis_closes_when_linear_regresses_one_dataset():
    results = _results(linear=1.01, matched=1.01)
    analysis = experiment.analyze(results)

    assert analysis["advancing_candidate"] is None
    assert not analysis["advances_to_selector_design"]
    assert analysis["recommendation"] == "close_linear_leaf_policy_route"


def test_partition_boundary_excludes_every_lockbox_task():
    boundary = experiment._partition_boundary()
    assert not set(experiment.TASKS).intersection(
        boundary["lockbox_task_ids"]
    )
    assert set(experiment.TASKS) <= set(boundary["confirmation_task_ids"])
