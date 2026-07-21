#!/usr/bin/env python3
"""Run and validate the M6 fast general development slice.

The runner intentionally produces descriptive, spent development evidence.
It cannot authorize a default or shipping claim.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import bench_compare_revisions as comparison
    from benchmark_adapters import DATASETS
    from campaign_lib.provenance import canonical_json_sha256, file_sha256
    from standing_evidence import (
        M6_DATASETS,
        M6_BACKTEST_COMPLETE,
        M6_BACKTEST_TERMINAL,
        M6_CONTRACT_FROZEN,
        M6_MODELS,
        M6_REPEAT,
        M6_SEED_COUNT,
        M6_SIZES,
        M6_SMOKE_DATASETS,
        M6_THREADS,
        M6_WEIGHT_MODES,
        contract_payload,
        m6_expected_grid,
    )
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks import bench_compare_revisions as comparison
    from benchmarks.benchmark_adapters import DATASETS
    from benchmarks.campaign_lib.provenance import (
        canonical_json_sha256,
        file_sha256,
    )
    from benchmarks.standing_evidence import (
        M6_DATASETS,
        M6_BACKTEST_COMPLETE,
        M6_BACKTEST_TERMINAL,
        M6_CONTRACT_FROZEN,
        M6_MODELS,
        M6_REPEAT,
        M6_SEED_COUNT,
        M6_SIZES,
        M6_SMOKE_DATASETS,
        M6_THREADS,
        M6_WEIGHT_MODES,
        contract_payload,
        m6_expected_grid,
    )


RUNNER_VERSION = "standing-evidence-runner-v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
_MECHANISM_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

_REGRESSION_METRIC_FIELDS = (
    "rmse",
    "mae",
    "r2",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
)
_CLASSIFICATION_METRIC_FIELDS = (
    "accuracy",
    "f1_macro",
    "log_loss",
    "brier",
    "weighted_accuracy",
    "weighted_f1_macro",
    "weighted_log_loss",
    "weighted_brier",
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    """Capture the committed identity and visible dirtiness of one source."""
    repository = repository.expanduser().resolve()
    if not (repository / "darkofit").is_dir():
        raise RuntimeError(f"not a DarkoFit source checkout: {repository}")
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repository:
        raise RuntimeError(
            f"source must name its Git root: {repository} (root is {top_level})"
        )
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "package_tree": _git(repository, "rev-parse", "HEAD:darkofit"),
        "branch": _git(repository, "branch", "--show-current"),
        "clean": not status,
        "status": status,
    }


def _require_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    label: str,
) -> None:
    if before != after:
        changed = sorted(
            key for key in before if before.get(key) != after.get(key)
        )
        raise RuntimeError(
            f"{label} source changed during M6 execution: {changed}"
        )


def validate_source_contract(
    harness: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    smoke: bool,
) -> None:
    """Enforce null-smoke and full-development source boundaries."""
    if not harness["clean"]:
        raise RuntimeError(
            "M6 requires a clean committed harness checkout"
        )
    if smoke:
        if control["path"] != candidate["path"]:
            raise RuntimeError(
                "M6 null smoke requires the same checkout for both arms"
            )
        if not control["clean"] or not candidate["clean"]:
            raise RuntimeError("M6 null smoke requires a clean committed checkout")
        return
    if not control["clean"] or not candidate["clean"]:
        raise RuntimeError("full M6 requires clean committed source checkouts")
    if control["package_tree"] == candidate["package_tree"]:
        raise RuntimeError(
            "full M6 requires a candidate package tree distinct from the "
            "control; "
            "use --smoke for a null comparison"
        )


def _row_identity(row: dict[str, str]) -> tuple[str, str, str, int, str]:
    try:
        seed = int(row["seed"])
        return (
            row["variant"],
            row["dataset"],
            row["size"],
            seed,
            row["weight_mode"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid M6 row identity: {row!r}") from exc


def _finite_value(
    row: dict[str, str],
    field: str,
    *,
    nonnegative: bool = False,
) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"M6 row has invalid {field!r}: {row.get(field)!r}"
        ) from exc
    invalid_sign = value < 0.0 if nonnegative else value <= 0.0
    if not math.isfinite(value) or invalid_sign:
        condition = "negative" if nonnegative else "non-positive"
        raise RuntimeError(
            f"M6 row has {condition} or non-finite {field!r}: {value!r}"
        )
    return value


def _finite_metric(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"M6 row has invalid metric {field!r}: {row.get(field)!r}"
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(
            f"M6 row has non-finite metric {field!r}: {value!r}"
        )
    return value


def validate_rows(
    rows: list[dict[str, str]],
    *,
    smoke: bool,
    control: Path,
    candidate: Path,
) -> dict[str, Any]:
    """Fail closed on incomplete, duplicated, failed, or misbound M6 rows."""
    expected = set(m6_expected_grid(smoke=smoke))
    identities = [_row_identity(row) for row in rows]
    counts = Counter(identities)
    duplicates = sorted(identity for identity, count in counts.items() if count != 1)
    actual = set(identities)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if duplicates or missing or unexpected or len(rows) != len(expected):
        raise RuntimeError(
            "M6 grid mismatch: "
            f"rows={len(rows)}, expected={len(expected)}, "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )

    expected_paths = {
        "control_default": str(control.expanduser().resolve()),
        "candidate_default": str(candidate.expanduser().resolve()),
    }
    failures = []
    for row in rows:
        identity = _row_identity(row)
        if row.get("status") != "ok":
            failures.append(
                {
                    "identity": identity,
                    "status": row.get("status"),
                    "error": row.get("error"),
                }
            )
            continue
        if row.get("revision_path") != expected_paths[row["variant"]]:
            raise RuntimeError(
                f"M6 row is bound to the wrong source: {identity}"
            )
        if row.get("task") != DATASETS[row["dataset"]].task:
            raise RuntimeError(f"M6 row task drifted: {identity}")
        if row.get("use_defaults") != "True":
            raise RuntimeError(f"M6 row did not use public defaults: {identity}")
        expected_primary = (
            "weighted_rmse"
            if row["task"] == "regression" and row["weight_mode"] == "stress"
            else (
                "rmse"
                if row["task"] == "regression"
                else (
                    "weighted_log_loss"
                    if row["weight_mode"] == "stress"
                    else "log_loss"
                )
            )
        )
        if row.get("primary_metric") != expected_primary:
            raise RuntimeError(f"M6 row primary metric drifted: {identity}")
        _finite_value(row, "fit_seconds")
        _finite_value(row, "predict_seconds")
        _finite_value(row, "worker_peak_rss_bytes")
        primary_value = _finite_value(
            row, "primary_value", nonnegative=True
        )
        metric_fields = (
            _REGRESSION_METRIC_FIELDS
            if row["task"] == "regression"
            else _CLASSIFICATION_METRIC_FIELDS
        )
        metric_values = {}
        for field in metric_fields:
            metric_values[field] = _finite_metric(row, field)
        if primary_value != metric_values[expected_primary]:
            raise RuntimeError(
                f"M6 row primary value disagrees with "
                f"{expected_primary!r}: {identity}"
            )
    if failures:
        raise RuntimeError(f"M6 worker failures: {failures}")
    return {
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "all_rows_ok": True,
        "grid_complete": True,
    }


def load_and_validate_csv(
    path: Path,
    *,
    smoke: bool,
    control: Path,
    candidate: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"M6 raw CSV is not a regular file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != comparison.CSV_FIELDS:
            raise RuntimeError("M6 raw CSV schema drifted")
        rows = list(reader)
    validation = validate_rows(
        rows,
        smoke=smoke,
        control=control,
        candidate=candidate,
    )
    return rows, validation


def _ratio_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def summarize_pairs(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize candidate/control loss and cost ratios from matched cells."""
    cells: dict[tuple[str, str, int, str], dict[str, dict[str, str]]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (
            row["dataset"],
            row["size"],
            int(row["seed"]),
            row["weight_mode"],
        )
        cells[key][row["variant"]] = row

    quality_ratios = []
    fit_ratios = []
    predict_ratios = []
    undefined_quality_ratios = 0
    wins = ties = losses = 0
    for key, pair in cells.items():
        if set(pair) != set(M6_MODELS):
            raise RuntimeError(f"incomplete M6 pair: {key}")
        control = pair["control_default"]
        candidate = pair["candidate_default"]
        if control["primary_metric"] != candidate["primary_metric"]:
            raise RuntimeError(f"primary metric drifted within M6 pair: {key}")
        control_loss = _finite_value(
            control, "primary_value", nonnegative=True
        )
        candidate_loss = _finite_value(
            candidate, "primary_value", nonnegative=True
        )
        if control_loss > 0.0:
            quality_ratios.append(candidate_loss / control_loss)
        elif candidate_loss == 0.0:
            quality_ratios.append(1.0)
        else:
            undefined_quality_ratios += 1
        fit_ratios.append(
            _finite_value(candidate, "fit_seconds")
            / _finite_value(control, "fit_seconds")
        )
        predict_ratios.append(
            _finite_value(candidate, "predict_seconds")
            / _finite_value(control, "predict_seconds")
        )
        if candidate_loss < control_loss:
            wins += 1
        elif candidate_loss > control_loss:
            losses += 1
        else:
            ties += 1
    return {
        "paired_cells": len(cells),
        "candidate_loss_wins": wins,
        "candidate_loss_ties": ties,
        "candidate_loss_losses": losses,
        "undefined_primary_loss_ratio_cells": undefined_quality_ratios,
        "candidate_over_control_primary_loss_ratio": _ratio_summary(
            quality_ratios
        ),
        "candidate_over_control_fit_seconds_ratio": _ratio_summary(fit_ratios),
        "candidate_over_control_predict_seconds_ratio": _ratio_summary(
            predict_ratios
        ),
    }


def hardware_fingerprint() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "numba", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


def _write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".manifest.json")


def _comparison_argv(
    *,
    control: Path,
    candidate: Path,
    csv_path: Path,
    smoke: bool,
    threads: int,
) -> list[str]:
    datasets = M6_SMOKE_DATASETS if smoke else M6_DATASETS
    seeds = 1 if smoke else M6_SEED_COUNT
    return [
        "--policy-suite",
        "standing-slice",
        "--control",
        str(control),
        "--candidate",
        str(candidate),
        "--datasets",
        *datasets,
        "--sizes",
        *M6_SIZES,
        "--seeds",
        str(seeds),
        "--repeat",
        str(M6_REPEAT),
        "--threads",
        str(threads),
        "--weight-modes",
        *M6_WEIGHT_MODES,
        "--models",
        *M6_MODELS,
        "--csv",
        str(csv_path),
    ]


def inspection_record(args: argparse.Namespace) -> dict[str, Any]:
    """Return the manifest record that makes repeated M6 use auditable."""
    mechanism_id = getattr(args, "mechanism_id", None)
    inspection_index = getattr(args, "inspection_index", None)
    if args.smoke:
        if mechanism_id is not None or inspection_index is not None:
            raise ValueError(
                "M6 null smoke does not accept mechanism inspection fields"
            )
        return {
            "counted": False,
            "mechanism_id": None,
            "inspection_index": None,
            "outcomes_spent": False,
        }
    if not isinstance(mechanism_id, str) or not _MECHANISM_ID_RE.fullmatch(
        mechanism_id
    ):
        raise ValueError(
            "full M6 requires --mechanism-id matching "
            "[a-z0-9][a-z0-9._-]{0,63}"
        )
    if (
        isinstance(inspection_index, bool)
        or not isinstance(inspection_index, int)
        or inspection_index < 1
    ):
        raise ValueError(
            "full M6 requires a positive --inspection-index"
        )
    return {
        "counted": True,
        "mechanism_id": mechanism_id,
        "inspection_index": inspection_index,
        "outcomes_spent": True,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=M6_THREADS)
    parser.add_argument(
        "--mechanism-id",
        help=(
            "stable lowercase id for the mechanism under a full M6 "
            "inspection"
        ),
    )
    parser.add_argument(
        "--inspection-index",
        type=int,
        help=(
            "one-based count of full M6 inspections for this mechanism; "
            "record every material run"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the null-comparison harness check; this is not candidate "
            "ranking evidence"
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if M6_BACKTEST_TERMINAL:
        raise RuntimeError(
            "M6 v3 is terminal and cannot execute again; use a new "
            "mechanism-specific contract built on paired-evidence-v1"
        )
    if args.threads != M6_THREADS:
        raise ValueError(
            f"the M6 contract requires exactly {M6_THREADS} threads"
        )
    inspection = inspection_record(args)
    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    output = args.csv.expanduser().absolute()
    manifest_path = _manifest_path(output)
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")

    harness_before = source_state(REPO_ROOT)
    control_before = source_state(control)
    candidate_before = source_state(candidate)
    validate_source_contract(
        harness_before,
        control_before,
        candidate_before,
        smoke=args.smoke,
    )

    with tempfile.TemporaryDirectory(prefix="darkofit-m6-") as temporary:
        temporary_csv = Path(temporary) / "m6_raw.csv"
        comparison_argv = _comparison_argv(
            control=control,
            candidate=candidate,
            csv_path=temporary_csv,
            smoke=args.smoke,
            threads=args.threads,
        )
        comparison.main(comparison_argv)
        rows, validation = load_and_validate_csv(
            temporary_csv,
            smoke=args.smoke,
            control=control,
            candidate=candidate,
        )
        harness_after = source_state(REPO_ROOT)
        control_after = source_state(control)
        candidate_after = source_state(candidate)
        _require_unchanged(harness_before, harness_after, label="harness")
        _require_unchanged(control_before, control_after, label="control")
        _require_unchanged(candidate_before, candidate_after, label="candidate")

        csv_bytes = temporary_csv.read_bytes()
        raw_sha256 = file_sha256(temporary_csv)
        contract = contract_payload()
        paired_summary = summarize_pairs(rows)
        if (
            args.smoke
            and paired_summary["candidate_loss_ties"]
            != paired_summary["paired_cells"]
        ):
            raise RuntimeError(
                "M6 null smoke changed primary loss between identical sources"
            )
        ranking_eligible = (
            not args.smoke
            and M6_CONTRACT_FROZEN
            and M6_BACKTEST_COMPLETE
        )
        manifest = {
            "runner_version": RUNNER_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_kind": (
                "harness_smoke"
                if args.smoke
                else (
                    "spent_development"
                    if ranking_eligible
                    else "contract_development"
                )
            ),
            "smoke": bool(args.smoke),
            "candidate_ranking_eligible": ranking_eligible,
            "shipping_or_default_claim_eligible": False,
            "inspection": inspection,
            "contract": contract,
            "contract_sha256": canonical_json_sha256(contract),
            "harness_source": harness_before,
            "control_source": control_before,
            "candidate_source": candidate_before,
            "hardware": hardware_fingerprint(),
            "execution": {
                "threads": args.threads,
                "datasets": list(
                    M6_SMOKE_DATASETS if args.smoke else M6_DATASETS
                ),
                "sizes": list(M6_SIZES),
                "seed_count": 1 if args.smoke else M6_SEED_COUNT,
                "weight_modes": list(M6_WEIGHT_MODES),
                "models": list(M6_MODELS),
                "repeat": M6_REPEAT,
                "variant_order": (
                    "parity_of_dataset_size_seed_block_plus_weight_index"
                ),
            },
            "raw_csv": {
                "path": str(output),
                "sha256": raw_sha256,
                **validation,
            },
            "paired_summary": paired_summary,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        wrote_output = False
        try:
            _write_create_only(output, csv_bytes)
            wrote_output = True
            _write_create_only(manifest_path, manifest_bytes)
        except BaseException:
            if wrote_output:
                output.unlink(missing_ok=True)
            raise

    print(f"wrote validated M6 rows to {output}")
    print(f"wrote M6 provenance manifest to {manifest_path}")
    return output, manifest_path


def main(argv: Optional[list[str]] = None) -> None:
    run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
