"""Summarize benchmark CSVs as default-policy regret reports.

The revision benchmark already records raw rows for datasets, seeds, weight
modes, policies, timings, and weighted metrics. This module adds the missing
decision layer: for each matched case, compare a designated default policy with
the best available policy and report quality regret plus speed/Pareto context.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CASE_COLUMNS = ("dataset", "task", "size", "seed", "weight_mode")
LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "log_loss",
    "brier",
    "weighted_rmse",
    "weighted_mae",
    "weighted_log_loss",
    "weighted_brier",
    "primary_value",
}
HIGHER_IS_BETTER = {
    "r2",
    "accuracy",
    "f1_macro",
    "weighted_r2",
    "weighted_accuracy",
    "weighted_f1_macro",
}


@dataclass(frozen=True)
class RegretCase:
    dataset: str
    task: str
    size: str
    seed: str
    weight_mode: str
    default_policy: str
    best_policy: str
    primary_metric: str
    default_value: float
    best_value: float
    regret_abs: float
    regret_pct: float
    default_fit_seconds: float
    best_fit_seconds: float
    fit_speed_ratio_vs_best: float
    default_predict_seconds: float
    best_predict_seconds: float
    predict_speed_ratio_vs_best: float
    pareto_dominated: bool
    dominators: str


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def metric_direction(metric: str) -> str:
    """Return ``lower`` or ``higher`` for a benchmark metric name."""
    metric = (metric or "").strip()
    if metric in LOWER_IS_BETTER or metric.endswith("_loss"):
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(f"unknown metric direction for {metric!r}")


def quality_better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "lower" else a > b


def quality_not_worse(a: float, b: float, direction: str) -> bool:
    return a <= b if direction == "lower" else a >= b


def quality_regret(default_value: float, best_value: float, direction: str) -> tuple[float, float]:
    """Return nonnegative absolute and percent regret for default vs best."""
    if direction == "lower":
        regret = default_value - best_value
    else:
        regret = best_value - default_value
    regret = max(0.0, regret)
    denom = max(abs(best_value), 1e-12)
    return regret, 100.0 * regret / denom


def _row_quality(row: dict[str, str]) -> tuple[str, float] | None:
    metric = (row.get("primary_metric") or "").strip()
    value = _to_float(row.get("primary_value"))
    if not metric or value is None:
        return None
    return metric, value


def _case_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in CASE_COLUMNS)


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def evaluate_default_regret(
    rows: Iterable[dict[str, str]],
    *,
    default_policy: str,
    reference_policies: set[str] | None = None,
) -> list[RegretCase]:
    """Compare one default policy with the best policy in every matched case."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status", "ok") not in {"", "ok"}:
            continue
        if _row_quality(row) is None:
            continue
        variant = row.get("variant") or row.get("model") or ""
        if variant:
            row = dict(row)
            row["variant"] = variant
        grouped.setdefault(_case_key(row), []).append(row)

    out: list[RegretCase] = []
    for key, case_rows in sorted(grouped.items()):
        default_rows = [row for row in case_rows if row.get("variant") == default_policy]
        if not default_rows:
            continue
        default_row = default_rows[0]
        primary = _row_quality(default_row)
        if primary is None:
            continue
        metric, default_value = primary
        direction = metric_direction(metric)

        candidates = [
            row for row in case_rows
            if reference_policies is None or row.get("variant") in reference_policies
        ]
        candidates = [
            row for row in candidates
            if _row_quality(row) is not None and _row_quality(row)[0] == metric
        ]
        if not candidates:
            continue
        best_row = min(
            candidates,
            key=lambda row: _row_quality(row)[1]
            if direction == "lower"
            else -_row_quality(row)[1],
        )
        best_value = _row_quality(best_row)[1]
        regret_abs, regret_pct = quality_regret(default_value, best_value, direction)

        default_fit = _to_float(default_row.get("fit_seconds")) or 0.0
        best_fit = _to_float(best_row.get("fit_seconds")) or 0.0
        default_predict = _to_float(default_row.get("predict_seconds")) or 0.0
        best_predict = _to_float(best_row.get("predict_seconds")) or 0.0

        dominators = []
        for row in candidates:
            if row is default_row:
                continue
            row_metric = _row_quality(row)
            row_fit = _to_float(row.get("fit_seconds"))
            if row_metric is None or row_fit is None:
                continue
            row_value = row_metric[1]
            value_ok = quality_not_worse(row_value, default_value, direction)
            fit_ok = row_fit <= default_fit
            strictly_better = (
                quality_better(row_value, default_value, direction)
                or row_fit < default_fit
            )
            if value_ok and fit_ok and strictly_better:
                dominators.append(row.get("variant", ""))

        out.append(
            RegretCase(
                dataset=key[0],
                task=key[1],
                size=key[2],
                seed=key[3],
                weight_mode=key[4],
                default_policy=default_policy,
                best_policy=best_row.get("variant", ""),
                primary_metric=metric,
                default_value=default_value,
                best_value=best_value,
                regret_abs=regret_abs,
                regret_pct=regret_pct,
                default_fit_seconds=default_fit,
                best_fit_seconds=best_fit,
                fit_speed_ratio_vs_best=(best_fit / default_fit if default_fit > 0 else float("nan")),
                default_predict_seconds=default_predict,
                best_predict_seconds=best_predict,
                predict_speed_ratio_vs_best=(
                    best_predict / default_predict if default_predict > 0 else float("nan")
                ),
                pareto_dominated=bool(dominators),
                dominators=";".join(sorted(set(dominators))),
            )
        )
    return out


def summarize(cases: list[RegretCase]) -> dict[str, float | int | str]:
    regrets = [case.regret_pct for case in cases]
    dominated = [case for case in cases if case.pareto_dominated]
    if not regrets:
        return {
            "cases": 0,
            "median_regret_pct": float("nan"),
            "p90_regret_pct": float("nan"),
            "worst_regret_pct": float("nan"),
            "worst_case": "",
            "pareto_dominated_cases": 0,
        }
    ordered = sorted(regrets)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    worst = max(cases, key=lambda case: case.regret_pct)
    return {
        "cases": len(cases),
        "median_regret_pct": statistics.median(regrets),
        "p90_regret_pct": ordered[p90_index],
        "worst_regret_pct": worst.regret_pct,
        "worst_case": f"{worst.dataset}/{worst.size}/seed={worst.seed}/weights={worst.weight_mode}",
        "pareto_dominated_cases": len(dominated),
    }


def write_cases_csv(path: Path, cases: list[RegretCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RegretCase.__dataclass_fields__)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: getattr(case, field) for field in fields})


def _print_report(cases: list[RegretCase], *, top: int) -> None:
    summary = summarize(cases)
    print("DEFAULT REGRET SUMMARY")
    print(f"cases: {summary['cases']}")
    print(f"median regret: {summary['median_regret_pct']:.3f}%")
    print(f"p90 regret: {summary['p90_regret_pct']:.3f}%")
    print(f"worst regret: {summary['worst_regret_pct']:.3f}%")
    print(f"worst case: {summary['worst_case']}")
    print(f"pareto dominated cases: {summary['pareto_dominated_cases']}")
    if not cases:
        return

    print()
    print("WORST CASES")
    print(
        "dataset                 task        size   seed weights  metric          "
        "default      best         regret% best_policy"
    )
    print("-" * 112)
    for case in sorted(cases, key=lambda item: item.regret_pct, reverse=True)[:top]:
        print(
            f"{case.dataset:23s} {case.task:11s} {case.size:6s} "
            f"{case.seed:4s} {case.weight_mode:8s} {case.primary_metric:14s} "
            f"{case.default_value:11.5g} {case.best_value:11.5g} "
            f"{case.regret_pct:8.3f} {case.best_policy}"
        )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path, help="raw benchmark CSV path(s)")
    parser.add_argument("--default-policy", default="candidate_default")
    parser.add_argument(
        "--reference-policy",
        action="append",
        dest="reference_policies",
        help="policy label to include as a reference; may be passed repeatedly",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    rows = read_rows(args.csv)
    cases = evaluate_default_regret(
        rows,
        default_policy=args.default_policy,
        reference_policies=set(args.reference_policies) if args.reference_policies else None,
    )
    _print_report(cases, top=args.top)
    if args.output_csv:
        write_cases_csv(args.output_csv, cases)
        print(f"\nwrote case rows to {args.output_csv}")


if __name__ == "__main__":
    main()
