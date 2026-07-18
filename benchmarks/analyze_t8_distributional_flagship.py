"""Validate and summarize the frozen T8 distributional flagship campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import stat
import statistics
import tempfile
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
MAX_ITERATIONS = 120
CSV_FIELDS = (
    "dataset",
    "model",
    "weight_mode",
    "seed",
    "n_train",
    "n_test",
    "n_features",
    "status",
    "reason",
    "data_sha256",
    "interval_method",
    "calibration_n",
    "fit_seconds",
    "predict_seconds",
    "best_iteration",
    "rmse_mu",
    "nll",
    "crps",
    "interval90_coverage",
    "interval90_width",
    "cov90_by_sigma",
    "sigma_mean",
    "sigma_min",
    "sigma_max",
)
EXPECTED_SHAPES = {
    "synthetic_100k": (100_000, 25_000, 6),
    "synthetic_t3_100k": (100_000, 25_000, 6),
    "openml_cpu_act": (6_144, 2_048, 21),
    "openml_wine_quality": (4_872, 1_625, 11),
    "openml_boston": (379, 127, 13),
}
EXPECTED_CONFORMAL_CALIBRATION_N = {
    "synthetic_100k": 10_000,
    "synthetic_t3_100k": 10_000,
    "openml_cpu_act": 614,
    "openml_wine_quality": 487,
    "openml_boston": 38,
}
ORIGINAL_ANALYZER_SHA256 = (
    "c8b52ee6313b7b3406648277aec53661f993ec4f53fe276201588044b31d4c0e"
)
FROZEN_RUNNER_SHA256 = (
    "382ba9059fcf430654748c0cc0c15427f42d9be98cf37aa3becd76d19f471d80"
)
CONFORMAL_MODEL = "darkofit_gaussian_es_conformal"
DISTRIBUTION_MODELS = frozenset(MODELS) - {"lightgbm_quantile_pair"}
INTERVAL_ONLY_MODEL = "lightgbm_quantile_pair"
ITERATION_MODELS = frozenset(
    {
        "darkofit_gaussian_es_calibrated",
        CONFORMAL_MODEL,
        "catboost_uncertainty",
    }
)
FROZEN_RAW = Path(__file__).resolve().with_name(
    "t8_distributional_flagship_raw.csv"
)
FROZEN_PROTOCOL = Path(__file__).resolve().with_name(
    "t8_distributional_flagship_protocol.md"
)
FROZEN_RAW_MARKDOWN = Path(__file__).resolve().with_name(
    "t8_distributional_flagship_raw.md"
)
RUNNER = Path(__file__).resolve().with_name("bench_distributional.py")
FROZEN_RAW_SHA256 = (
    "eedf6e037a3ef1e6628fdf2a2ae1c46bf8fae09df93b6566dd3c8b22dba75578"
)
FROZEN_PROTOCOL_SHA256 = (
    "210e574d4c9d562febb43f95e7169ed3f52a009605c4741c7ae415a94f78b84e"
)
FROZEN_RAW_MARKDOWN_SHA256 = (
    "5f7bf6da078898b47864791c9a689cf67316c033b74fc2d9d3517a4ea1dc8dd2"
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name(
    "t8_distributional_flagship_result.md"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(row, name, *, optional=False):
    value = row[name]
    if optional and value == "":
        return None
    if value == "":
        raise ValueError(f"missing {name} in {row}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name} in {row}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite {name} in {row}")
    return number


def _integer(row, name):
    value = row[name]
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer {name} in {row}") from error
    if value != str(number):
        raise ValueError(f"noncanonical integer {name} in {row}")
    if not -(2**63) <= number <= 2**63 - 1:
        raise ValueError(f"out-of-range integer {name} in {row}")
    return number


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _coverage_bins(value, key):
    pieces = value.split("/")
    if len(pieces) != 5:
        raise ValueError(f"invalid conditional coverage at {key}")
    values = [float(piece) for piece in pieces]
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values):
        raise ValueError(f"invalid conditional coverage at {key}")
    return values


def _geomean(values):
    values = tuple(float(value) for value in values)
    if not values or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _same_destination(left, right):
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _reject_symlink_directory(path, message):
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    for component in (absolute, *absolute.parents):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {component}")


def _create_owned_directories(path, message):
    missing = []
    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing.append(current)
            current = current.parent
            continue
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {current}")
        break
    owned = []
    try:
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                mode = directory.lstat().st_mode
                if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                    raise RuntimeError(f"{message}: {directory}")
                continue
            identity = directory.lstat()
            if not stat.S_ISDIR(identity.st_mode):
                raise RuntimeError(f"{message}: {directory}")
            owned.append(
                (directory, (identity.st_dev, identity.st_ino))
            )
    except BaseException:
        _remove_owned_directories(owned)
        raise
    return owned


def _remove_owned_directories(owned):
    for directory, expected in reversed(owned):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == expected
            ):
                directory.rmdir()
        except OSError:
            pass


def _verify_published_identity(path, expected, message):
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{message}: {path}") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise RuntimeError(f"{message}: {path}")


def _validate_output_path(output, raw_path):
    if output.is_symlink():
        raise RuntimeError(f"refusing symlink T8 output: {output}")
    _reject_symlink_directory(
        output.parent, "refusing symlink T8 output directory"
    )
    protected = (
        raw_path,
        FROZEN_RAW,
        FROZEN_PROTOCOL,
        FROZEN_RAW_MARKDOWN,
        RUNNER,
    )
    if any(_same_destination(output, path) for path in protected):
        raise RuntimeError(f"refusing protected T8 output: {output}")
    if output.exists():
        raise RuntimeError(f"refusing existing T8 output: {output}")


def _unlink_if_owned(path, identity):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        path.unlink()


def _atomic_write_text(path, text):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing T8 output: {path}")
    message = "refusing symlink T8 output directory"
    _reject_symlink_directory(path.parent, message)
    owned_directories = _create_owned_directories(path.parent, message)
    temporary = None
    published_identity = None
    try:
        _reject_symlink_directory(path.parent, message)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(text.encode())
            handle.flush()
            os.fsync(handle.fileno())
            identity = os.fstat(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing existing T8 output: {path}"
            ) from error
        published_identity = (identity.st_dev, identity.st_ino)
        _verify_published_identity(
            path,
            published_identity,
            "T8 output publish identity changed",
        )
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published_identity is not None:
            try:
                _unlink_if_owned(path, published_identity)
            except OSError:
                pass
        _remove_owned_directories(owned_directories)
        raise
    try:
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            _unlink_if_owned(path, published_identity)
        except OSError:
            pass
        _remove_owned_directories(owned_directories)
        raise


def _evidence_snapshot():
    analyzer_sha256 = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()
    runner_sha256 = _sha256(RUNNER)
    protocol_sha256 = _sha256(FROZEN_PROTOCOL)
    raw_markdown_sha256 = _sha256(FROZEN_RAW_MARKDOWN)
    if runner_sha256 != FROZEN_RUNNER_SHA256:
        raise RuntimeError("T8 frozen runner changed")
    if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("T8 frozen protocol changed")
    if raw_markdown_sha256 != FROZEN_RAW_MARKDOWN_SHA256:
        raise RuntimeError("T8 frozen raw markdown changed")
    return (
        analyzer_sha256,
        runner_sha256,
        protocol_sha256,
        raw_markdown_sha256,
    )


def _load(path, *, with_sha256=False):
    raw_bytes = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with io.StringIO(raw_bytes.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("T8 CSV schema changed")
        rows = list(reader)
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
    }
    expected_order = [
        (dataset, model, seed)
        for dataset in DATASETS
        for seed in SEEDS
        for model in MODELS
    ]
    observed = set()
    fingerprints = defaultdict(set)
    dimensions = defaultdict(set)
    for row in rows:
        if set(row) != set(CSV_FIELDS) or any(
            not isinstance(row.get(name), str) for name in CSV_FIELDS
        ):
            raise ValueError("T8 CSV row shape changed")
        seed = _integer(row, "seed")
        key = (row["dataset"], row["model"], seed)
        if key in observed:
            raise ValueError(f"duplicate campaign coordinate: {key}")
        observed.add(key)
        if row["status"] != "ok":
            raise ValueError(f"non-ok campaign coordinate {key}: {row['reason']}")
        expected_reason = (
            "interval-only baseline"
            if row["model"] == INTERVAL_ONLY_MODEL
            else ""
        )
        if row["reason"] != expected_reason:
            raise ValueError(f"unexpected reason at {key}")
        if row["weight_mode"] != "none":
            raise ValueError(f"unexpected weight mode at {key}")
        fingerprint = row["data_sha256"]
        if not _is_sha256(fingerprint):
            raise ValueError(f"invalid data fingerprint at {key}")
        fingerprints[(row["dataset"], seed)].add(fingerprint)
        shape = tuple(
            _integer(row, name)
            for name in ("n_train", "n_test", "n_features")
        )
        if shape != EXPECTED_SHAPES.get(row["dataset"]):
            raise ValueError(f"invalid data shape at {key}")
        dimensions[(row["dataset"], seed)].add(shape)
        coverage = _number(row, "interval90_coverage")
        width = _number(row, "interval90_width")
        fit_seconds = _number(row, "fit_seconds")
        predict_seconds = _number(row, "predict_seconds")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"invalid coverage at {key}")
        if width <= 0.0:
            raise ValueError(f"invalid interval width at {key}")
        if fit_seconds <= 0.0 or predict_seconds <= 0.0:
            raise ValueError(f"invalid timing at {key}")
        if row["model"] in ITERATION_MODELS:
            try:
                best_iteration = _integer(row, "best_iteration")
            except ValueError as error:
                raise ValueError(f"invalid best iteration at {key}") from error
            if (
                best_iteration < 1
                or best_iteration > MAX_ITERATIONS
                or (
                    row["model"] == "catboost_uncertainty"
                    and best_iteration != MAX_ITERATIONS
                )
            ):
                raise ValueError(f"invalid best iteration at {key}")
        elif row["best_iteration"] != "":
            raise ValueError(f"unexpected best iteration at {key}")
        if row["model"] == CONFORMAL_MODEL:
            if row["interval_method"] != "split_conformal":
                raise ValueError(f"wrong conformal method at {key}")
            try:
                calibration_n = _integer(row, "calibration_n")
            except ValueError as error:
                raise ValueError(
                    f"invalid conformal calibration size at {key}"
                ) from error
            if (
                calibration_n
                != EXPECTED_CONFORMAL_CALIBRATION_N[row["dataset"]]
            ):
                raise ValueError(
                    f"invalid conformal calibration size at {key}"
                )
        elif row["model"] == "lightgbm_quantile_pair":
            if row["interval_method"] != "quantile":
                raise ValueError(f"wrong quantile method at {key}")
        elif row["interval_method"] != "parametric":
            raise ValueError(f"wrong parametric method at {key}")
        elif row["calibration_n"] != "":
            raise ValueError(f"unexpected calibration set at {key}")
        if row["model"] in DISTRIBUTION_MODELS:
            rmse = _number(row, "rmse_mu")
            _number(row, "nll")
            crps = _number(row, "crps")
            sigma_mean = _number(row, "sigma_mean")
            sigma_min = _number(row, "sigma_min")
            sigma_max = _number(row, "sigma_max")
            if rmse < 0.0 or crps < 0.0:
                raise ValueError(f"invalid distribution metric at {key}")
            if (
                sigma_min <= 0.0
                or not sigma_min <= sigma_mean <= sigma_max
            ):
                raise ValueError(f"invalid distribution scale at {key}")
            coverage_bins = _coverage_bins(row["cov90_by_sigma"], key)
            bin_size, remainder = divmod(shape[1], len(coverage_bins))
            binned_coverage = sum(
                value * (bin_size + (index < remainder))
                for index, value in enumerate(coverage_bins)
            ) / shape[1]
            if not math.isclose(
                binned_coverage,
                coverage,
                rel_tol=0.0,
                abs_tol=0.00051,
            ):
                raise ValueError(
                    f"inconsistent conditional coverage at {key}"
                )
        elif row["model"] == INTERVAL_ONLY_MODEL:
            forbidden = (
                "calibration_n",
                "rmse_mu",
                "nll",
                "crps",
                "cov90_by_sigma",
                "sigma_mean",
                "sigma_min",
                "sigma_max",
            )
            if any(row[name] != "" for name in forbidden):
                raise ValueError(
                    f"interval-only row reports distribution metrics at {key}"
                )
        else:  # pragma: no cover - coordinate boundary rejects this first.
            raise ValueError(f"unknown model at {key}")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"coordinate mismatch; missing={missing}, extra={extra}")
    observed_order = [
        (row["dataset"], row["model"], _integer(row, "seed"))
        for row in rows
    ]
    if observed_order != expected_order:
        raise ValueError("T8 campaign row order changed")
    inconsistent = {
        key: values for key, values in fingerprints.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"models did not share identical data: {inconsistent}")
    inconsistent_shapes = {
        key: values for key, values in dimensions.items() if len(values) != 1
    }
    if inconsistent_shapes:
        raise ValueError(
            f"models did not share identical data shapes: {inconsistent_shapes}"
        )
    if raw_sha256 != FROZEN_RAW_SHA256:
        raise ValueError("T8 frozen raw content changed")
    if with_sha256:
        return rows, raw_sha256
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


def _render(
    rows,
    raw_path=None,
    *,
    raw_sha256=None,
    analyzer_sha256=None,
    runner_sha256=None,
    protocol_sha256=None,
    raw_markdown_sha256=None,
):
    if raw_sha256 is None:
        if raw_path is None:
            raise ValueError("raw path or SHA-256 is required")
        snapshot_rows, raw_sha256 = _load(raw_path, with_sha256=True)
        if snapshot_rows != rows:
            raise ValueError("T8 rows do not match the raw content snapshot")
    if (
        analyzer_sha256 is None
        or runner_sha256 is None
        or protocol_sha256 is None
        or raw_markdown_sha256 is None
    ):
        (
            analyzer_sha256,
            runner_sha256,
            protocol_sha256,
            raw_markdown_sha256,
        ) = _evidence_snapshot()
    if (
        runner_sha256 != FROZEN_RUNNER_SHA256
        or protocol_sha256 != FROZEN_PROTOCOL_SHA256
        or raw_markdown_sha256 != FROZEN_RAW_MARKDOWN_SHA256
    ):
        raise RuntimeError("T8 evidence snapshot changed")
    summaries = _summarize(rows)
    lines = [
        "# T8 distributional flagship result",
        "",
        f"- Raw CSV SHA-256: `{raw_sha256}`",
        f"- Raw run report SHA-256: `{raw_markdown_sha256}`",
        f"- Frozen protocol SHA-256: `{protocol_sha256}`",
        f"- Original run-time runner SHA-256: `{FROZEN_RUNNER_SHA256}`",
        f"- Current runner SHA-256: `{runner_sha256}`",
        f"- Original run-time analyzer SHA-256: `{ORIGINAL_ANALYZER_SHA256}`",
        f"- Current hardened analyzer SHA-256: `{analyzer_sha256}`",
        f"- Complete coordinates: **{len(rows)}/{len(DATASETS) * len(MODELS) * len(SEEDS)}**",
        "- Status: descriptive Tier-E evidence; no default or automatic policy was tested.",
        "- Post-run amendment: the current analyzer hardens validation and "
        "report layout only; the immutable raw coordinates and metrics were "
        "not rerun or changed.",
        "",
        "Coverage and width are adjacent columns in every result row. They are "
        "intentionally not collapsed into a single score.",
        "",
        "## Per-dataset 90% interval results",
        "",
        "| Dataset | Model | Coverage | Width | Absolute gap | NLL | CRPS | Fit s | Predict s |",
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
                        _fmt(summary["width"]),
                        _fmt(abs(summary["coverage"] - TARGET_COVERAGE)),
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
        "| Model | Mean coverage | Geomean width / conformal | "
        "Mean absolute coverage gap | Worst cell coverage | Best cell coverage |",
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
            f"{_fmt(_geomean(width_ratios))} | "
            f"{_fmt(statistics.mean(abs(value - TARGET_COVERAGE) for value in coverages))} | "
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=FROZEN_RAW,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args(argv)
    args.csv = Path(os.path.abspath(os.path.expanduser(args.csv)))
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    return args


def main(argv=None):
    args = parse_args(argv)
    _validate_output_path(args.output, args.csv)
    (
        analyzer_sha256,
        runner_sha256,
        protocol_sha256,
        raw_markdown_sha256,
    ) = _evidence_snapshot()
    rows, raw_sha256 = _load(args.csv, with_sha256=True)
    report = _render(
        rows,
        raw_sha256=raw_sha256,
        analyzer_sha256=analyzer_sha256,
        runner_sha256=runner_sha256,
        protocol_sha256=protocol_sha256,
        raw_markdown_sha256=raw_markdown_sha256,
    )
    _atomic_write_text(args.output, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
