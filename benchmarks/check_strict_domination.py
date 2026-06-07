"""Check whether candidate catboost strictly dominates upstream rows.

The checker is intentionally about the benchmark contract, not about model
training. It consumes raw revision-comparison CSV rows and reports concrete
blocking failures: missing rows, error rows, quality regressions, timing
regressions, and semantic non-equivalence in the upstream-compatible lane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


KEY_FIELDS = (
    "dataset",
    "size",
    "split_mode",
    "weight_mode",
    "ensemble_size",
    "seed",
)


def _as_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _repeat_values(row, field):
    value = row.get(field)
    if not value:
        return []
    out = []
    for part in value.split(";"):
        parsed = _as_float(part)
        if parsed is not None:
            out.append(parsed)
    return out


def _fit_seconds(row, stat):
    if stat == "min":
        return _as_float(row.get("fit_seconds"))
    values = _repeat_values(row, "fit_repeat_seconds")
    if not values:
        return _as_float(row.get("fit_seconds"))
    values = sorted(values)
    if stat == "median":
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return 0.5 * (values[mid - 1] + values[mid])
    if stat == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unknown fit-time statistic {stat!r}")


def _row_key(row):
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def _key_dict(key):
    return dict(zip(KEY_FIELDS, key))


def _is_weighted(row):
    metric = row.get("primary_metric", "")
    return row.get("weight_mode", "") != "none" or metric.startswith("weighted_")


def _quality_tolerance(upstream_value, weighted):
    scale = abs(upstream_value)
    if weighted:
        return max(1e-6, 1e-3 * scale)
    return max(1e-10, 1e-6 * scale)


def _fit_ratio_limit(upstream_fit):
    return 1.05 if upstream_fit < 0.20 else 1.03


def _timing_control_limits(rows, *, baseline, candidate):
    """Return row/aggregate timing-noise envelopes from same-code controls."""
    return _timing_control_limits_for_stat(
        rows, baseline=baseline, candidate=candidate, fit_time_stat="min")


def _timing_control_limits_for_stat(
    rows, *, baseline, candidate, fit_time_stat):
    """Return row/aggregate timing-noise envelopes from same-code controls."""
    grouped = defaultdict(dict)
    ratios = []
    limits = {}
    for row in rows:
        variant = row.get("variant")
        if variant not in {baseline, candidate}:
            continue
        grouped[_row_key(row)][variant] = row

    for key, pair in grouped.items():
        base = pair.get(baseline)
        cand = pair.get(candidate)
        if base is None or cand is None:
            continue
        if base.get("status") != "ok" or cand.get("status") != "ok":
            continue
        base_fit = _fit_seconds(base, fit_time_stat)
        cand_fit = _fit_seconds(cand, fit_time_stat)
        if base_fit in (None, 0) or cand_fit in (None, 0):
            continue
        ratio = cand_fit / base_fit
        ratios.append(ratio)
        # Same-revision labels are arbitrary, so treat a fast "candidate" row
        # as evidence of symmetric timing noise in the other direction too.
        limits[key] = max(limits.get(key, 1.0), ratio, 1.0 / ratio)

    aggregate = None
    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        aggregate = max(geomean, 1.0 / geomean)
    return limits, aggregate


def _failure(kind, key, message, **extra):
    out = {"kind": kind, "key": _key_dict(key), "message": message}
    out.update(extra)
    return out


def load_rows(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def evaluate_rows(
    rows,
    *,
    baseline="upstream_matched",
    candidate="candidate_catboost",
    mode="upstream-compatible",
    aggregate_fit_ratio_limit=1.0,
    timing_control_limits=None,
    aggregate_timing_control_limit=None,
    fit_time_stat="min",
):
    """Return a strict-domination report dictionary for raw benchmark rows."""
    if mode not in {"upstream-compatible", "product"}:
        raise ValueError(f"unknown strict-domination mode {mode!r}")

    grouped = defaultdict(dict)
    failures = []
    for row in rows:
        variant = row.get("variant")
        if variant not in {baseline, candidate}:
            continue
        key = _row_key(row)
        grouped[key][variant] = row

    fit_ratios = []
    compared = 0
    for key in sorted(grouped):
        pair = grouped[key]
        base = pair.get(baseline)
        cand = pair.get(candidate)
        if base is None:
            failures.append(_failure(
                "missing_row", key, f"missing baseline row {baseline!r}"))
            continue
        if cand is None:
            failures.append(_failure(
                "missing_row", key, f"missing candidate row {candidate!r}"))
            continue

        for label, row in ((baseline, base), (candidate, cand)):
            if row.get("status") != "ok":
                failures.append(_failure(
                    "error_row",
                    key,
                    f"{label} row has status {row.get('status')!r}",
                    variant=label,
                    error=row.get("error", ""),
                ))
        if base.get("status") != "ok" or cand.get("status") != "ok":
            continue

        weighted = _is_weighted(cand)
        policy = cand.get("validation_weight_policy", "")
        if weighted and mode == "upstream-compatible" and policy != mode:
            failures.append(_failure(
                "semantic_non_equivalence",
                key,
                "weighted candidate row was not run with upstream-compatible "
                "validation semantics",
                validation_weight_policy=policy,
            ))
            continue
        if weighted and mode == "product" and policy != mode:
            failures.append(_failure(
                "semantic_non_equivalence",
                key,
                "weighted candidate row was not run with product validation "
                "semantics",
                validation_weight_policy=policy,
            ))
            continue

        base_metric = _as_float(base.get("primary_value"))
        cand_metric = _as_float(cand.get("primary_value"))
        base_fit = _fit_seconds(base, fit_time_stat)
        cand_fit = _fit_seconds(cand, fit_time_stat)
        if None in (base_metric, cand_metric, base_fit, cand_fit):
            failures.append(_failure(
                "missing_row",
                key,
                "row is missing primary metric or fit_seconds",
            ))
            continue

        compared += 1
        quality_tol = _quality_tolerance(base_metric, weighted)
        quality_delta = cand_metric - base_metric
        if quality_delta > quality_tol:
            failures.append(_failure(
                "quality_regression",
                key,
                "candidate primary metric is materially worse",
                upstream_primary=base_metric,
                candidate_primary=cand_metric,
                tolerance=quality_tol,
                primary_metric=cand.get("primary_metric", ""),
            ))

        fit_ratio = cand_fit / base_fit if base_fit > 0 else math.inf
        fit_ratios.append(fit_ratio)
        if mode == "product" and weighted:
            denominator = max(abs(base_metric), 1e-12)
            improvement = (base_metric - cand_metric) / denominator
            fit_limit = 1.05 if improvement >= 0.002 else 1.03
        else:
            fit_limit = _fit_ratio_limit(base_fit)
        if timing_control_limits:
            fit_limit = max(fit_limit, timing_control_limits.get(key, fit_limit))
        if fit_ratio > fit_limit:
            failures.append(_failure(
                "timing_regression",
                key,
                "candidate fit time is materially slower",
                upstream_fit_seconds=base_fit,
                candidate_fit_seconds=cand_fit,
                fit_ratio=fit_ratio,
                limit=fit_limit,
            ))

    geomean = None
    if fit_ratios:
        geomean = math.exp(sum(math.log(r) for r in fit_ratios) / len(fit_ratios))
        aggregate_limit = aggregate_fit_ratio_limit
        if aggregate_timing_control_limit is not None:
            aggregate_limit = max(aggregate_limit, aggregate_timing_control_limit)
        if geomean > aggregate_limit:
            failures.append({
                "kind": "aggregate_timing_regression",
                "key": {},
                "message": "geometric mean fit ratio exceeds the strict gate",
                "geomean_fit_ratio": geomean,
                "limit": aggregate_limit,
            })

    return {
        "baseline": baseline,
        "candidate": candidate,
        "mode": mode,
        "fit_time_stat": fit_time_stat,
        "key_fields": KEY_FIELDS,
        "n_compared": compared,
        "geomean_fit_ratio": geomean,
        "aggregate_fit_ratio_limit": aggregate_fit_ratio_limit,
        "aggregate_timing_control_limit": aggregate_timing_control_limit,
        "timing_control_rows": 0 if not timing_control_limits else len(timing_control_limits),
        "passed": not failures,
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--baseline", default="upstream_matched")
    parser.add_argument("--candidate", default="candidate_catboost")
    parser.add_argument(
        "--mode",
        choices=["upstream-compatible", "product"],
        default="upstream-compatible",
    )
    parser.add_argument(
        "--aggregate-fit-ratio-limit",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--fit-time-stat",
        choices=["min", "median", "mean"],
        default="min",
        help=(
            "which statistic from fit_repeat_seconds to compare. The default "
            "keeps the accepted min-of-repeat strict gate; median/mean are "
            "diagnostic robustness checks."
        ),
    )
    parser.add_argument(
        "--timing-control",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional same-revision raw CSV. Row timing limits are widened to "
            "the observed same-code timing-noise envelope for matching keys."
        ),
    )
    parser.add_argument("--timing-control-baseline", default="upstream_matched")
    parser.add_argument("--timing-control-candidate", default="candidate_matched")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("strict_domination_report.json"),
    )
    args = parser.parse_args(argv)

    timing_limits = None
    aggregate_timing_limit = None
    if args.timing_control:
        control_rows = []
        for path in args.timing_control:
            control_rows.extend(load_rows(path))
        timing_limits, aggregate_timing_limit = _timing_control_limits_for_stat(
            control_rows,
            baseline=args.timing_control_baseline,
            candidate=args.timing_control_candidate,
            fit_time_stat=args.fit_time_stat,
        )

    report = evaluate_rows(
        load_rows(args.csv),
        baseline=args.baseline,
        candidate=args.candidate,
        mode=args.mode,
        aggregate_fit_ratio_limit=args.aggregate_fit_ratio_limit,
        timing_control_limits=timing_limits,
        aggregate_timing_control_limit=aggregate_timing_limit,
        fit_time_stat=args.fit_time_stat,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"strict_domination passed={report['passed']} "
        f"n_compared={report['n_compared']} "
        f"failures={len(report['failures'])} "
        f"geomean_fit_ratio={report['geomean_fit_ratio']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
