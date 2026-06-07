"""Summarize repeat-distribution timing for strict-domination failures.

The strict checker uses min-of-repeat timing, which is good for excluding one
slow trial but can overreact when two revisions have different timing tails. This
helper reads a raw revision-comparison CSV plus a strict-domination report and
prints/writes per-failure min/median/mean repeat ratios. Optional same-revision
controls add a matching timing-noise envelope for each key.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


KEY_FIELDS = (
    "dataset",
    "size",
    "split_mode",
    "weight_mode",
    "ensemble_size",
    "seed",
)


OUT_FIELDS = [
    "dataset",
    "size",
    "split_mode",
    "weight_mode",
    "ensemble_size",
    "seed",
    "kind",
    "min_ratio",
    "median_ratio",
    "mean_ratio",
    "candidate_min",
    "baseline_min",
    "candidate_median",
    "baseline_median",
    "candidate_mean",
    "baseline_mean",
    "control_min_envelope",
    "control_median_envelope",
    "control_mean_envelope",
    "median_over_control",
    "mean_over_control",
]


def _key(row):
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def _key_from_report(key_dict):
    return tuple(str(key_dict.get(field, "")) for field in KEY_FIELDS)


def _read_rows(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _repeat_values(row, field="fit_repeat_seconds"):
    text = row.get(field, "")
    if not text:
        value = row.get("fit_seconds", "")
        return np.array([float(value)], dtype=np.float64)
    return np.array([float(part) for part in text.split(";")], dtype=np.float64)


def _ratio_summary(candidate, baseline):
    cand = _repeat_values(candidate)
    base = _repeat_values(baseline)
    return {
        "min_ratio": float(cand.min() / base.min()),
        "median_ratio": float(np.median(cand) / np.median(base)),
        "mean_ratio": float(cand.mean() / base.mean()),
        "candidate_min": float(cand.min()),
        "baseline_min": float(base.min()),
        "candidate_median": float(np.median(cand)),
        "baseline_median": float(np.median(base)),
        "candidate_mean": float(cand.mean()),
        "baseline_mean": float(base.mean()),
    }


def _control_envelopes(paths, baseline, candidate):
    by_key = defaultdict(dict)
    for path in paths:
        for row in _read_rows(path):
            variant = row.get("variant")
            if variant in {baseline, candidate} and row.get("status") == "ok":
                by_key[_key(row)][variant] = row

    envelopes = {}
    for key, pair in by_key.items():
        base = pair.get(baseline)
        cand = pair.get(candidate)
        if base is None or cand is None:
            continue
        ratios = _ratio_summary(cand, base)
        envelopes[key] = {
            "control_min_envelope": max(
                ratios["min_ratio"], 1.0 / ratios["min_ratio"]
            ),
            "control_median_envelope": max(
                ratios["median_ratio"], 1.0 / ratios["median_ratio"]
            ),
            "control_mean_envelope": max(
                ratios["mean_ratio"], 1.0 / ratios["mean_ratio"]
            ),
        }
    return envelopes


def summarize(csv_path, report_path, *, baseline, candidate,
              control_paths=(), control_baseline=None, control_candidate=None):
    rows = _read_rows(csv_path)
    by_key = defaultdict(dict)
    for row in rows:
        if row.get("variant") in {baseline, candidate} and row.get("status") == "ok":
            by_key[_key(row)][row["variant"]] = row

    controls = _control_envelopes(
        control_paths,
        control_baseline or baseline,
        control_candidate or candidate,
    )
    report = json.loads(Path(report_path).read_text())
    out = []
    for failure in report.get("failures", []):
        if failure.get("kind") != "timing_regression":
            continue
        key = _key_from_report(failure.get("key", {}))
        pair = by_key.get(key, {})
        base = pair.get(baseline)
        cand = pair.get(candidate)
        if base is None or cand is None:
            continue
        row = dict(zip(KEY_FIELDS, key))
        row["kind"] = failure.get("kind", "")
        row.update(_ratio_summary(cand, base))
        row.update({
            "control_min_envelope": "",
            "control_median_envelope": "",
            "control_mean_envelope": "",
            "median_over_control": "",
            "mean_over_control": "",
        })
        control = controls.get(key)
        if control:
            row.update(control)
            median_env = control["control_median_envelope"]
            mean_env = control["control_mean_envelope"]
            row["median_over_control"] = (
                "" if not math.isfinite(median_env) or median_env == 0.0
                else row["median_ratio"] / median_env
            )
            row["mean_over_control"] = (
                "" if not math.isfinite(mean_env) or mean_env == 0.0
                else row["mean_ratio"] / mean_env
            )
        out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", default="upstream_matched")
    parser.add_argument("--candidate", default="candidate_catboost")
    parser.add_argument("--timing-control", type=Path, action="append", default=[])
    parser.add_argument("--timing-control-baseline", default=None)
    parser.add_argument("--timing-control-candidate", default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows = summarize(
        args.csv,
        args.report,
        baseline=args.baseline,
        candidate=args.candidate,
        control_paths=args.timing_control,
        control_baseline=args.timing_control_baseline,
        control_candidate=args.timing_control_candidate,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in OUT_FIELDS}
                             for row in rows)
        print(f"wrote {len(rows)} failure rows to {args.out}")
    else:
        writer = csv.DictWriter(
            __import__("sys").stdout, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in OUT_FIELDS}
                         for row in rows)


if __name__ == "__main__":
    main()
