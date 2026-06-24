"""Tests for the benchmark default-regret report."""

import csv
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from default_regret_report import (  # noqa: E402
    evaluate_default_regret,
    metric_direction,
    read_rows,
    summarize,
    write_cases_csv,
)


def _row(
    variant,
    dataset,
    task,
    seed,
    primary_metric,
    primary_value,
    fit_seconds,
    predict_seconds=0.01,
    weight_mode="none",
):
    return {
        "status": "ok",
        "variant": variant,
        "dataset": dataset,
        "task": task,
        "size": "tiny",
        "seed": str(seed),
        "weight_mode": weight_mode,
        "primary_metric": primary_metric,
        "primary_value": str(primary_value),
        "fit_seconds": str(fit_seconds),
        "predict_seconds": str(predict_seconds),
    }


def test_metric_direction_knows_benchmark_metrics():
    assert metric_direction("rmse") == "lower"
    assert metric_direction("weighted_log_loss") == "lower"
    assert metric_direction("f1_macro") == "higher"
    assert metric_direction("weighted_accuracy") == "higher"

    with pytest.raises(ValueError, match="unknown metric"):
        metric_direction("mystery")


def test_evaluate_default_regret_uses_best_quality_reference():
    rows = [
        _row("candidate_default", "friedman", "regression", 0, "rmse", 10.0, 2.0),
        _row("candidate_catboost_explicit", "friedman", "regression", 0, "rmse", 8.0, 3.0),
        _row("candidate_lightgbm_explicit", "friedman", "regression", 0, "rmse", 11.0, 1.0),
        _row("candidate_default", "binary", "binary", 0, "weighted_log_loss", 0.3, 1.0),
        _row("candidate_catboost_explicit", "binary", "binary", 0, "weighted_log_loss", 0.25, 0.8),
    ]

    cases = evaluate_default_regret(rows, default_policy="candidate_default")

    assert len(cases) == 2
    friedman = [case for case in cases if case.dataset == "friedman"][0]
    assert friedman.best_policy == "candidate_catboost_explicit"
    assert friedman.regret_abs == pytest.approx(2.0)
    assert friedman.regret_pct == pytest.approx(25.0)
    assert friedman.fit_speed_ratio_vs_best == pytest.approx(1.5)
    assert friedman.pareto_dominated is False

    binary = [case for case in cases if case.dataset == "binary"][0]
    assert binary.pareto_dominated is True
    assert binary.dominators == "candidate_catboost_explicit"


def test_evaluate_default_regret_handles_higher_is_better_metrics():
    rows = [
        _row("default", "cls", "binary", 0, "f1_macro", 0.80, 1.0),
        _row("better", "cls", "binary", 0, "f1_macro", 0.84, 1.1),
    ]

    cases = evaluate_default_regret(rows, default_policy="default")

    assert cases[0].best_policy == "better"
    assert cases[0].regret_abs == pytest.approx(0.04)
    assert cases[0].regret_pct == pytest.approx(100.0 * 0.04 / 0.84)


def test_summary_and_csv_round_trip(tmp_path):
    rows = [
        _row("candidate_default", "a", "regression", 0, "rmse", 10.0, 1.0),
        _row("candidate_best", "a", "regression", 0, "rmse", 9.0, 1.0),
        _row("candidate_default", "b", "regression", 0, "rmse", 5.0, 1.0),
        _row("candidate_best", "b", "regression", 0, "rmse", 5.0, 1.0),
    ]
    cases = evaluate_default_regret(rows, default_policy="candidate_default")

    report = summarize(cases)

    assert report["cases"] == 2
    assert report["worst_case"] == "a/tiny/seed=0/weights=none"
    assert report["worst_regret_pct"] == pytest.approx(100.0 / 9.0)

    path = tmp_path / "default_regret.csv"
    write_cases_csv(path, cases)
    loaded = read_rows([path])

    assert len(loaded) == 2
    assert loaded[0]["default_policy"] == "candidate_default"


def test_read_rows_accepts_multiple_csvs(tmp_path):
    fields = ["status", "variant", "dataset", "task", "size", "seed", "weight_mode", "primary_metric", "primary_value"]
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for idx, path in enumerate(paths):
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "status": "ok",
                    "variant": f"v{idx}",
                    "dataset": "d",
                    "task": "regression",
                    "size": "tiny",
                    "seed": str(idx),
                    "weight_mode": "none",
                    "primary_metric": "rmse",
                    "primary_value": "1.0",
                }
            )

    assert [row["variant"] for row in read_rows(paths)] == ["v0", "v1"]
