"""Run the calibrated catboost strict-domination gate.

This orchestrates the accepted timing contract in one short command:

1. candidate catboost vs upstream matched rows;
2. same-code upstream vs upstream control rows;
3. calibrated strict-domination report using min-of-repeats by default;
4. optional median/mean diagnostic reports.

It deliberately shells out to the existing raw benchmark and checker scripts so
their behavior stays single-source-of-truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _script_path(name):
    return Path(__file__).resolve().with_name(name)


def _add_if_present(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def _add_many(cmd, flag, values):
    if values:
        cmd.append(flag)
        cmd.extend(str(value) for value in values)


def _compare_command(args, *, candidate_path, models, csv_path):
    cmd = [
        sys.executable,
        str(_script_path("bench_compare_revisions.py")),
        "--upstream",
        str(args.upstream),
        "--candidate",
        str(candidate_path),
        "--models",
        *models,
        "--iterations",
        str(args.iterations),
        "--patience",
        str(args.patience),
        "--threads",
        str(args.threads),
        "--repeat",
        str(args.repeat),
        "--seeds",
        str(args.seeds),
        "--depth",
        str(args.depth),
        "--validation-weight-policy",
        args.validation_weight_policy,
        "--csv",
        str(csv_path),
    ]
    _add_many(cmd, "--sizes", args.sizes)
    _add_many(cmd, "--datasets", args.datasets)
    _add_many(cmd, "--weight-modes", args.weight_modes)
    _add_many(cmd, "--split-modes", args.split_modes)
    _add_many(cmd, "--ensemble-sizes", args.ensemble_sizes)
    _add_if_present(cmd, "--case-manifest", args.case_manifest)
    _add_if_present(cmd, "--learning-rate", args.learning_rate)
    _add_if_present(cmd, "--max-bins-ts", args.max_bins_ts)
    if args.gc_between_repeats:
        cmd.append("--gc-between-repeats")
    if args.verbose_timing:
        cmd.append("--verbose-timing")
    if args.weighted_target_stats:
        cmd.append("--weighted-target-stats")
    if args.ordered_boosting:
        cmd.append("--ordered-boosting")
    return cmd


def _checker_command(args, *, raw_csv, control_csv, report_path, fit_time_stat):
    return [
        sys.executable,
        str(_script_path("check_strict_domination.py")),
        str(raw_csv),
        "--baseline",
        "upstream_matched",
        "--candidate",
        args.candidate_label,
        "--mode",
        args.validation_weight_policy,
        "--fit-time-stat",
        fit_time_stat,
        "--timing-control",
        str(control_csv),
        "--timing-control-baseline",
        "upstream_matched",
        "--timing-control-candidate",
        "candidate_matched",
        "--out",
        str(report_path),
    ]


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--candidate-label", default="candidate_catboost")
    parser.add_argument("--sizes", nargs="+", default=["medium"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["friedman_numeric", "numeric_binary", "categorical_binary"],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-bins-ts", type=int, default=None)
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=None,
        help="optional blocker manifest passed through to the raw benchmark",
    )
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=["none", "uniform", "stress"],
        default=["none", "stress"],
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["row", "group"],
        default=["row"],
    )
    parser.add_argument(
        "--ensemble-sizes",
        nargs="+",
        type=int,
        default=[1],
    )
    parser.add_argument(
        "--validation-weight-policy",
        choices=["upstream-compatible", "product"],
        default="upstream-compatible",
    )
    parser.add_argument("--gc-between-repeats", action="store_true")
    parser.add_argument(
        "--verbose-timing",
        action="store_true",
        help=(
            "pass through per-phase timing collection to the raw benchmark; "
            "intended for diagnosis, not as the accepted timing gate itself"
        ),
    )
    parser.add_argument("--weighted-target-stats", action="store_true")
    parser.add_argument("--ordered-boosting", action="store_true")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("benchmarks/catboost_calibrated_gate"),
    )
    parser.add_argument(
        "--diagnostic-fit-time-stats",
        nargs="*",
        choices=["median", "mean"],
        default=["median"],
        help=(
            "extra calibrated diagnostic reports. The accepted strict report "
            "always uses min-of-repeats."
        ),
    )
    return parser.parse_args(argv)


def _run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _run_gate(cmd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw_csv = out_prefix.with_name(out_prefix.name + "_raw.csv")
    control_csv = out_prefix.with_name(out_prefix.name + "_same_upstream.csv")
    report_min = out_prefix.with_name(out_prefix.name + "_calibrated_min_report.json")

    _run(_compare_command(
        args,
        candidate_path=args.candidate,
        models=["upstream_matched", args.candidate_label],
        csv_path=raw_csv,
    ))
    _run(_compare_command(
        args,
        candidate_path=args.upstream,
        models=["upstream_matched", "candidate_matched"],
        csv_path=control_csv,
    ))
    exit_code = _run_gate(_checker_command(
        args,
        raw_csv=raw_csv,
        control_csv=control_csv,
        report_path=report_min,
        fit_time_stat="min",
    ))

    for stat in args.diagnostic_fit_time_stats:
        report = out_prefix.with_name(
            out_prefix.name + f"_calibrated_{stat}_diagnostic_report.json")
        cmd = _checker_command(
            args,
            raw_csv=raw_csv,
            control_csv=control_csv,
            report_path=report,
            fit_time_stat=stat,
        )
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)

    print(f"wrote raw rows to {raw_csv}")
    print(f"wrote same-code control rows to {control_csv}")
    print(f"wrote accepted calibrated min report to {report_min}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
