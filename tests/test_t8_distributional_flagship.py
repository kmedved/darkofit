import statistics
from pathlib import Path

import pytest

from benchmarks import analyze_t8_distributional_flagship as analysis


def test_t8_raw_boundary_and_shared_fingerprints_reproduce():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    rows = analysis._load(raw)
    assert len(rows) == 75
    assert {
        (row["dataset"], row["model"], int(row["seed"]))
        for row in rows
    } == {
        (dataset, model, seed)
        for dataset in analysis.DATASETS
        for model in analysis.MODELS
        for seed in analysis.SEEDS
    }


def test_t8_result_is_deterministically_regenerated():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    result = Path("benchmarks/t8_distributional_flagship_result.md")
    rows = analysis._load(raw)
    assert result.read_text(encoding="utf-8") == analysis._render(rows, raw)


def test_t8_conformal_coverage_width_summary_is_exact():
    raw = Path("benchmarks/t8_distributional_flagship_raw.csv")
    summaries = analysis._summarize(analysis._load(raw))
    conformal = [
        summaries[(dataset, analysis.CONFORMAL_MODEL)]
        for dataset in analysis.DATASETS
    ]
    parametric = [
        summaries[(dataset, "darkofit_gaussian_es_calibrated")]
        for dataset in analysis.DATASETS
    ]
    conformal_gap = statistics.mean(
        abs(row["coverage"] - 0.9) for row in conformal
    )
    parametric_gap = statistics.mean(
        abs(row["coverage"] - 0.9) for row in parametric
    )
    width_ratio = analysis._geomean(
        left["width"] / right["width"]
        for left, right in zip(conformal, parametric)
    )
    assert conformal_gap == pytest.approx(0.010966164458409079)
    assert parametric_gap == pytest.approx(0.012935432181253769)
    assert width_ratio == pytest.approx(0.9830526057278705)
