"""Summarize raw revision-comparison benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _as_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _ratio(value, base):
    if value is None or base in (None, 0):
        return None
    return value / base


def load_rows(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["dataset"],
            row["size"],
            row.get("split_mode", ""),
            row["weight_mode"],
            row.get("ensemble_size", ""),
            row["variant"],
        )
        grouped[key].append(row)

    out = {}
    for key, vals in grouped.items():
        primary_metric = vals[0].get("primary_metric", "")
        out[key] = {
            "n": len(vals),
            "task": vals[0].get("task", ""),
            "primary_metric": primary_metric,
            "primary_value": _mean(_as_float(v.get("primary_value")) for v in vals),
            "fit_seconds": _mean(_as_float(v.get("fit_seconds")) for v in vals),
            "predict_seconds": _mean(_as_float(v.get("predict_seconds")) for v in vals),
            "best_iteration": _mean(_as_float(v.get("best_iteration")) for v in vals),
        }
    return out


def print_summary(summary, baseline):
    groups = sorted({key[:5] for key in summary})
    print(
        "dataset,size,split_mode,weight_mode,ensemble_size,variant,n,"
        "primary_metric,primary_value,metric_vs_base,fit_seconds,fit_vs_base,"
        "best_iteration,iter_vs_base"
    )
    for dataset, size, split_mode, weight_mode, ensemble_size in groups:
        base_key = (dataset, size, split_mode, weight_mode,
                    ensemble_size, baseline)
        base = summary.get(base_key)
        base_metric = None if base is None else base["primary_value"]
        base_fit = None if base is None else base["fit_seconds"]
        base_iter = None if base is None else base["best_iteration"]
        prefix = (dataset, size, split_mode, weight_mode, ensemble_size)
        variants = sorted(key[5] for key in summary if key[:5] == prefix)
        for variant in variants:
            row = summary[(*prefix, variant)]
            print(
                ",".join(
                    [
                        dataset,
                        size,
                        split_mode,
                        weight_mode,
                        ensemble_size,
                        variant,
                        str(row["n"]),
                        row["primary_metric"],
                        _fmt(row["primary_value"]),
                        _fmt(_ratio(row["primary_value"], base_metric), 3),
                        _fmt(row["fit_seconds"], 3),
                        _fmt(_ratio(row["fit_seconds"], base_fit), 3),
                        _fmt(row["best_iteration"], 1),
                        _fmt(_ratio(row["best_iteration"], base_iter), 3),
                    ]
                )
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--baseline", default="upstream_matched")
    args = parser.parse_args(argv)
    print_summary(aggregate(load_rows(args.csv)), args.baseline)


if __name__ == "__main__":
    main()
