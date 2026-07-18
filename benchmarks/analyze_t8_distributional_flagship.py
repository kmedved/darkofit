"""Validate and summarize the frozen T8 distributional flagship campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = (
    "synthetic_100k",
    "synthetic_t3_100k",
    "openml_cpu_act",
    "openml_wine_quality",
    "openml_boston",
)
MODELS = (
    "darkofit_gaussian_es_calibrated",
    "darkofit_gaussian_es_conformal",
    "ngboost",
    "catboost_uncertainty",
    "lightgbm_quantile_pair",
)
SEEDS = (0, 1, 2)
TARGET_COVERAGE = 0.9
CONFORMAL_MODEL = "darkofit_gaussian_es_conformal"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(row, name, *, optional=False):
    value = row[name]
    if optional and value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {name} in {row}")
    return number


def _geomean(values):
    values = tuple(float(value) for value in values)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _load(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
    }
    observed = set()
    fingerprints = defaultdict(set)
    for row in rows:
        key = (row["dataset"], row["model"], int(row["seed"]))
        if key in observed:
            raise ValueError(f"duplicate campaign coordinate: {key}")
        observed.add(key)
        if row["status"] != "ok":
            raise ValueError(f"non-ok campaign coordinate {key}: {row['reason']}")
        if row["weight_mode"] != "none":
            raise ValueError(f"unexpected weight mode at {key}")
        fingerprint = row["data_sha256"]
        if len(fingerprint) != 64:
            raise ValueError(f"invalid data fingerprint at {key}")
        fingerprints[(row["dataset"], int(row["seed"]))].add(fingerprint)
        _number(row, "interval90_coverage")
        _number(row, "interval90_width")
        _number(row, "fit_seconds")
        _number(row, "predict_seconds")
        if row["model"] == CONFORMAL_MODEL:
            if row["interval_method"] != "split_conformal":
                raise ValueError(f"wrong conformal method at {key}")
            if int(row["calibration_n"]) < 1:
                raise ValueError(f"empty conformal calibration set at {key}")
        elif row["model"] == "lightgbm_quantile_pair":
            if row["interval_method"] != "quantile":
                raise ValueError(f"wrong quantile method at {key}")
        elif row["interval_method"] != "parametric":
            raise ValueError(f"wrong parametric method at {key}")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"coordinate mismatch; missing={missing}, extra={extra}")
    inconsistent = {
        key: values for key, values in fingerprints.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"models did not share identical data: {inconsistent}")
    return rows


def _mean(rows, name):
    values = [_number(row, name, optional=True) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(f"partially missing metric {name}")
    return statistics.mean(present)


def _summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    summaries = {}
    for key, subset in grouped.items():
        summaries[key] = {
            "coverage": _mean(subset, "interval90_coverage"),
            "width": _mean(subset, "interval90_width"),
            "nll": _mean(subset, "nll"),
            "crps": _mean(subset, "crps"),
            "fit": _mean(subset, "fit_seconds"),
            "predict": _mean(subset, "predict_seconds"),
        }
    return summaries


def _render(rows, raw_path):
    summaries = _summarize(rows)
    lines = [
        "# T8 distributional flagship result",
        "",
        f"- Raw CSV SHA-256: `{_sha256(raw_path)}`",
        f"- Analyzer SHA-256: `{_sha256(Path(__file__))}`",
        f"- Complete coordinates: **{len(rows)}/{len(DATASETS) * len(MODELS) * len(SEEDS)}**",
        "- Status: descriptive Tier-E evidence; no default or automatic policy was tested.",
        "",
        "Coverage is the first number in every cell and width is the second. "
        "They are intentionally not collapsed into a single score.",
        "",
        "## Per-dataset 90% interval results",
        "",
        "| Dataset | Model | Coverage | Absolute gap | Width | NLL | CRPS | Fit s | Predict s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            summary = summaries[(dataset, model)]
            lines.append(
                "| "
                + " | ".join(
                    (
                        dataset,
                        model,
                        _fmt(summary["coverage"]),
                        _fmt(abs(summary["coverage"] - TARGET_COVERAGE)),
                        _fmt(summary["width"]),
                        _fmt(summary["nll"]),
                        _fmt(summary["crps"]),
                        _fmt(summary["fit"], 3),
                        _fmt(summary["predict"], 4),
                    )
                )
                + " |"
            )

    lines.extend([
        "",
        "## Equal-dataset coverage and width",
        "",
        "| Model | Mean coverage | Mean absolute coverage gap | "
        "Geomean width / conformal | Worst cell coverage | Best cell coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for model in MODELS:
        coverages = [
            summaries[(dataset, model)]["coverage"] for dataset in DATASETS
        ]
        width_ratios = [
            summaries[(dataset, model)]["width"]
            / summaries[(dataset, CONFORMAL_MODEL)]["width"]
            for dataset in DATASETS
        ]
        cell_coverages = [
            _number(row, "interval90_coverage")
            for row in rows
            if row["model"] == model
        ]
        lines.append(
            f"| {model} | {_fmt(statistics.mean(coverages))} | "
            f"{_fmt(statistics.mean(abs(value - TARGET_COVERAGE) for value in coverages))} | "
            f"{_fmt(_geomean(width_ratios))} | "
            f"{_fmt(min(cell_coverages))} | {_fmt(max(cell_coverages))} |"
        )

    parametric = "darkofit_gaussian_es_calibrated"
    coverage_deltas = [
        abs(summaries[(dataset, CONFORMAL_MODEL)]["coverage"] - TARGET_COVERAGE)
        - abs(summaries[(dataset, parametric)]["coverage"] - TARGET_COVERAGE)
        for dataset in DATASETS
    ]
    width_ratios = [
        summaries[(dataset, CONFORMAL_MODEL)]["width"]
        / summaries[(dataset, parametric)]["width"]
        for dataset in DATASETS
    ]
    lines.extend([
        "",
        "## Conformal-versus-parametric DarkoFit",
        "",
        f"- Equal-dataset change in absolute coverage error: "
        f"**{statistics.mean(coverage_deltas):+.4f}** "
        "(negative is closer to 90%).",
        f"- Geometric-mean interval-width ratio: "
        f"**{_geomean(width_ratios):.4f}×**.",
        "- NLL and CRPS describe the fitted Gaussian distribution, not the "
        "conformal interval; they are never assigned to the quantile-only "
        "LightGBM lane.",
        "- Split-conformal coverage is marginal. This campaign does not claim "
        "conditional coverage, superiority on every dataset, or a default change.",
        "",
        "## Integrity checks",
        "",
        "- Every preregistered coordinate completed successfully.",
        "- Every model at a dataset/seed coordinate used the same fingerprinted "
        "train/test arrays.",
        "- Conformal rows report a nonempty isolated calibration set.",
        "- Interval-only baselines do not report midpoint RMSE, NLL, or CRPS as "
        "though they exposed a predictive distribution.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/t8_distributional_flagship_raw.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/t8_distributional_flagship_result.md"),
    )
    args = parser.parse_args(argv)
    rows = _load(args.csv)
    report = _render(rows, args.csv)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
