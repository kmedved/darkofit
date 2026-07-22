#!/usr/bin/env python3
"""Verify and analyze the frozen v0.11 private ensemble evidence artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import run_v011_ensemble_evidence as campaign
except ImportError:  # direct script execution
    import run_v011_ensemble_evidence as campaign


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "benchmarks" / "v011_ensemble_evidence_raw.json"
DEFAULT_RESULT = ROOT / "benchmarks" / "v011_ensemble_evidence_result.json"
DEFAULT_NOTE = ROOT / "benchmarks" / "v011_ensemble_evidence_result.md"


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise RuntimeError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def _dispersion(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != campaign.BLOCKS or not np.all(np.isfinite(array)):
        raise RuntimeError("paired repeat series is incomplete")
    med = float(np.median(array))
    q25, q75 = np.percentile(array, [25.0, 75.0])
    return {
        "series": [float(value) for value in array],
        "median": med,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "iqr": float(q75 - q25),
        "iqr_over_median": float((q75 - q25) / med) if med else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_raw(raw: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    contract = campaign.load_contract(contract_path)
    if (
        raw.get("schema_version") != 1
        or raw.get("contract_id") != campaign.CONTRACT_ID
        or raw.get("status") != "complete"
        or raw.get("execution") != campaign.execution_spec()
        or raw.get("claims") != campaign.claim_spec()
        or raw.get("reproduction_checked") is not True
        or raw.get("contract", {}).get("sha256") != campaign.sha256(contract_path)
        or raw.get("contract", {}).get("bindings") != contract["bindings"]
    ):
        raise RuntimeError("raw v0.11 ensemble evidence envelope is invalid")
    rows = raw.get("rows")
    expected_count = campaign.BLOCKS * (
        len(campaign.QUALITY_CASES) * len(campaign.QUALITY_ARMS)
        + len(campaign.PREDICTION_CASES) * len(campaign.PREDICTION_ARMS)
    )
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError("raw v0.11 ensemble evidence row count is invalid")
    expected_keys = set()
    for block in range(campaign.BLOCKS):
        for case_id in campaign.QUALITY_CASES:
            for position, arm in enumerate(campaign.quality_order(case_id, block)):
                expected_keys.add(("quality", block, case_id, arm, position))
        for case_id in campaign.PREDICTION_CASES:
            for position, arm in enumerate(campaign.prediction_order(case_id, block)):
                expected_keys.add(("prediction", block, case_id, arm, position))
    actual_keys = {
        (
            row.get("kind"),
            row.get("block"),
            row.get("case_id"),
            row.get("arm"),
            row.get("position"),
        )
        for row in rows
    }
    if actual_keys != expected_keys or len(actual_keys) != len(rows):
        raise RuntimeError("raw v0.11 ensemble evidence grid is invalid")
    manifests = contract["case_manifests"]
    for row in rows:
        manifest = manifests[row["case_id"]]
        for name, expected in manifest["fingerprints"].items():
            if row.get(name) != expected:
                raise RuntimeError(f"row fingerprint drifted: {row['case_id']}/{name}")
        if row.get("fit_rows") != manifest["fit_rows"]:
            raise RuntimeError(f"fit row count drifted: {row['case_id']}")
        if row.get("worker_stdout") is not None or row.get("worker_stderr") is not None:
            raise RuntimeError("formal worker emitted unexpected output")
        if row["kind"] == "quality":
            if (
                not math.isfinite(float(row["primary_loss"]))
                or float(row["primary_loss"]) <= 0
                or not math.isfinite(float(row["fit_seconds"]))
                or float(row["fit_seconds"]) <= 0
                or row["archive"].get("roundtrip_exact") is not True
                or row["fit_rss"].get("scope") != "worker_plus_recursive_children"
                or row["fit_rss"].get("errors")
                or int(row["fit_rss"].get("samples", 0)) < 2
            ):
                raise RuntimeError("quality/cost row failed integrity checks")
        else:
            predictions = row.get("predictions")
            if not isinstance(predictions, Mapping) or set(predictions) != {
                str(value) for value in campaign.BATCH_SIZES
            }:
                raise RuntimeError("prediction timing grid is incomplete")
            for record in predictions.values():
                if (
                    record.get("minimum_interval_met") is not True
                    or len(record.get("pilot_seconds", ())) != campaign.PILOT_CALLS
                    or int(record.get("calls", 0)) < campaign.MIN_CALLS
                    or int(record.get("calls", 0)) > campaign.MAX_CALLS
                    or float(record.get("interval_seconds", 0.0))
                    < campaign.MIN_INTERVAL_SECONDS
                    or not math.isfinite(float(record.get("seconds_per_call", math.nan)))
                    or float(record["seconds_per_call"]) <= 0
                ):
                    raise RuntimeError("prediction timing row failed integrity checks")
    return contract


def _rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        (row["kind"], row["block"], row["case_id"], row["arm"]): row
        for row in rows
    }


def _reproduction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = _rows_by_key(rows)
    expected = campaign.immutable_ratios()
    per_case: dict[str, Any] = {}
    for case_id in campaign.QUALITY_CASES:
        series = []
        for block in range(campaign.BLOCKS):
            single = by_key[("quality", block, case_id, campaign.DARKO_SINGLE)]
            candidate = by_key[("quality", block, case_id, campaign.DARKO_V3)]
            series.append(float(candidate["primary_loss"]) / float(single["primary_loss"]))
        differences = [abs(value - expected["per_case"][case_id]) for value in series]
        per_case[case_id] = {
            "expected_ratio": expected["per_case"][case_id],
            "current_ratio": _dispersion(series),
            "maximum_absolute_difference": max(differences),
            "within_frozen_tolerance": max(differences)
            <= campaign.REPRODUCTION_ABS_TOLERANCE,
        }
    aggregate = {}
    for name, cases in {
        "pooled": campaign.QUALITY_CASES,
        "sports": tuple(
            case for case in campaign.QUALITY_CASES if case.startswith("sports_")
        ),
        "general": tuple(
            case for case in campaign.QUALITY_CASES if case.startswith("general_")
        ),
    }.items():
        series = [
            _geomean(
                per_case[case]["current_ratio"]["series"][block] for case in cases
            )
            for block in range(campaign.BLOCKS)
        ]
        aggregate[name] = {
            "expected_ratio": expected[name],
            "current_ratio": _dispersion(series),
            "maximum_absolute_difference": max(abs(value - expected[name]) for value in series),
            "within_frozen_tolerance": max(abs(value - expected[name]) for value in series)
            <= campaign.REPRODUCTION_ABS_TOLERANCE,
        }
    passed = all(value["within_frozen_tolerance"] for value in per_case.values()) and all(
        value["within_frozen_tolerance"] for value in aggregate.values()
    )
    if not passed:
        raise RuntimeError("completed raw artifact does not satisfy reproduction contract")
    return {
        "absolute_ratio_tolerance": campaign.REPRODUCTION_ABS_TOLERANCE,
        "passed": True,
        "per_case": per_case,
        "aggregate": aggregate,
    }


def _quality_uncertainty(reproduction: Mapping[str, Any]) -> dict[str, Any]:
    ratios = {
        case_id: float(record["current_ratio"]["median"])
        for case_id, record in reproduction["per_case"].items()
    }
    season_ratios = {}
    for season in campaign.m3b.SPORTS_SEASONS:
        season_ratios[str(season)] = _geomean(
            ratios[case_id]
            for case_id in campaign.QUALITY_CASES
            if case_id.startswith(f"sports_{season}_")
        )
    sports_values = np.asarray(list(season_ratios.values()), dtype=np.float64)
    sports_rng = np.random.default_rng(campaign.SPORTS_BOOTSTRAP_SEED)
    sports_draws = np.empty(campaign.SPORTS_BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, campaign.SPORTS_BOOTSTRAP_DRAWS, 10_000):
        count = min(10_000, campaign.SPORTS_BOOTSTRAP_DRAWS - start)
        indices = sports_rng.integers(0, len(sports_values), size=(count, len(sports_values)))
        sports_draws[start : start + count] = np.exp(
            np.mean(np.log(sports_values[indices]), axis=1)
        )
    sports_loo = {
        f"omit_{season}": _geomean(
            value for key, value in season_ratios.items() if key != str(season)
        )
        for season in campaign.m3b.SPORTS_SEASONS
    }
    general_ratios = {
        case_id: ratios[case_id] for case_id in campaign.PREDICTION_CASES
    }
    general_values = np.asarray(list(general_ratios.values()), dtype=np.float64)
    general_rng = np.random.default_rng(campaign.GENERAL_BOOTSTRAP_SEED)
    general_draws = np.empty(campaign.GENERAL_BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, campaign.GENERAL_BOOTSTRAP_DRAWS, 10_000):
        count = min(10_000, campaign.GENERAL_BOOTSTRAP_DRAWS - start)
        indices = general_rng.integers(0, len(general_values), size=(count, len(general_values)))
        general_draws[start : start + count] = np.exp(
            np.mean(np.log(general_values[indices]), axis=1)
        )
    general_loo = {
        f"omit_{case_id}": _geomean(
            value for key, value in general_ratios.items() if key != case_id
        )
        for case_id in general_ratios
    }
    return {
        "sports": {
            "scope": "three season clusters; three targets geometrically pooled within season",
            "season_ratios": season_ratios,
            "point_ratio": _geomean(sports_values),
            "bootstrap_draws": campaign.SPORTS_BOOTSTRAP_DRAWS,
            "bootstrap_seed": campaign.SPORTS_BOOTSTRAP_SEED,
            "bootstrap_percentiles": dict(
                zip(
                    ("p2_5", "p50", "p97_5"),
                    map(float, np.percentile(sports_draws, [2.5, 50.0, 97.5])),
                )
            ),
            "leave_one_season_out": sports_loo,
        },
        "general": {
            "scope": "four fixed seeded 75/25 cases; not independent population draws",
            "case_ratios": general_ratios,
            "point_ratio": _geomean(general_values),
            "sample_std_log_ratio": float(np.std(np.log(general_values), ddof=1)),
            "bootstrap_draws": campaign.GENERAL_BOOTSTRAP_DRAWS,
            "bootstrap_seed": campaign.GENERAL_BOOTSTRAP_SEED,
            "bootstrap_percentiles": dict(
                zip(
                    ("p2_5", "p50", "p97_5"),
                    map(float, np.percentile(general_draws, [2.5, 50.0, 97.5])),
                )
            ),
            "leave_one_case_out": general_loo,
        },
    }


def _paired_costs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = _rows_by_key(rows)
    comparisons = {
        "v3_vs_single": (campaign.DARKO_V3, campaign.DARKO_SINGLE),
        "v3_vs_existing_bootstrap8": (campaign.DARKO_V3, campaign.DARKO_BOOTSTRAP),
    }
    metrics = {
        "fit_seconds": lambda row: float(row["fit_seconds"]),
        "peak_rss_bytes": lambda row: float(row["fit_rss"]["peak_bytes"]),
        "peak_rss_delta_bytes": lambda row: float(row["fit_rss"]["peak_delta_bytes"]),
        "archive_bytes": lambda row: float(row["archive"]["bytes"]),
    }
    result = {}
    for comparison, (numerator_arm, denominator_arm) in comparisons.items():
        per_case = {}
        for case_id in campaign.QUALITY_CASES:
            per_metric = {}
            for metric, getter in metrics.items():
                numerator = []
                denominator = []
                ratios = []
                for block in range(campaign.BLOCKS):
                    left = getter(by_key[("quality", block, case_id, numerator_arm)])
                    right = getter(by_key[("quality", block, case_id, denominator_arm)])
                    numerator.append(left)
                    denominator.append(right)
                    ratios.append(left / right if right > 0 else math.inf)
                if not np.all(np.isfinite(ratios)):
                    raise RuntimeError("cost ratio has a zero or non-finite denominator")
                per_metric[metric] = {
                    "ratio": _dispersion(ratios),
                    "numerator_absolute": _dispersion(numerator),
                    "denominator_absolute": _dispersion(denominator),
                }
            per_case[case_id] = per_metric
        aggregate = {
            metric: _geomean(
                per_case[case_id][metric]["ratio"]["median"]
                for case_id in campaign.QUALITY_CASES
            )
            for metric in metrics
        }
        result[comparison] = {"per_case": per_case, "equal_case_geomean_ratio": aggregate}
    return result


def _prediction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = _rows_by_key(rows)
    pairs = {
        "darkofit_single_vs_chimeraboost_single": (
            campaign.DARKO_SINGLE,
            campaign.CHIMERA_SINGLE,
        ),
        "darkofit_single_vs_catboost_single": (
            campaign.DARKO_SINGLE,
            campaign.CATBOOST_SINGLE,
        ),
        "chimeraboost_single_vs_catboost_single": (
            campaign.CHIMERA_SINGLE,
            campaign.CATBOOST_SINGLE,
        ),
        "darkofit_v3_vs_darkofit_single": (campaign.DARKO_V3, campaign.DARKO_SINGLE),
        "darkofit_v3_vs_chimeraboost_ensemble8": (
            campaign.DARKO_V3,
            campaign.CHIMERA_ENSEMBLE,
        ),
    }
    result = {}
    for name, (numerator_arm, denominator_arm) in pairs.items():
        coordinates = {}
        for case_id in campaign.PREDICTION_CASES:
            for batch in campaign.BATCH_SIZES:
                ratios = []
                numerator_seconds = []
                denominator_seconds = []
                for block in range(campaign.BLOCKS):
                    left = by_key[("prediction", block, case_id, numerator_arm)][
                        "predictions"
                    ][str(batch)]
                    right = by_key[("prediction", block, case_id, denominator_arm)][
                        "predictions"
                    ][str(batch)]
                    left_seconds = float(left["seconds_per_call"])
                    right_seconds = float(right["seconds_per_call"])
                    ratios.append(left_seconds / right_seconds)
                    numerator_seconds.append(left_seconds)
                    denominator_seconds.append(right_seconds)
                coordinates[f"{case_id}:{batch}"] = {
                    "case_id": case_id,
                    "batch_rows": batch,
                    "seconds_ratio": _dispersion(ratios),
                    "numerator_seconds": _dispersion(numerator_seconds),
                    "denominator_seconds": _dispersion(denominator_seconds),
                }
        medians = [value["seconds_ratio"]["median"] for value in coordinates.values()]
        result[name] = {
            "numerator_arm": numerator_arm,
            "denominator_arm": denominator_arm,
            "coordinates": coordinates,
            "equal_coordinate_geomean_seconds_ratio": _geomean(medians),
            "coordinates_numerator_faster": sum(value < 1.0 for value in medians),
            "coordinates_tied": sum(value == 1.0 for value in medians),
            "coordinates_numerator_slower": sum(value > 1.0 for value in medians),
        }
    absolute = defaultdict(dict)
    for arm in campaign.PREDICTION_ARMS:
        for case_id in campaign.PREDICTION_CASES:
            for batch in campaign.BATCH_SIZES:
                series = [
                    float(
                        by_key[("prediction", block, case_id, arm)]["predictions"][
                            str(batch)
                        ]["rows_per_second"]
                    )
                    for block in range(campaign.BLOCKS)
                ]
                absolute[arm][f"{case_id}:{batch}"] = _dispersion(series)
    return {"comparisons": result, "absolute_rows_per_second": dict(absolute)}


def analyze(raw_path: Path, contract_path: Path) -> dict[str, Any]:
    raw = _load_json(raw_path)
    _validate_raw(raw, contract_path)
    rows = raw["rows"]
    reproduction = _reproduction(rows)
    result = {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "raw": {"path": str(raw_path.resolve()), "sha256": campaign.sha256(raw_path)},
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": campaign.sha256(contract_path),
        },
        "sources": raw["sources"],
        "machine": raw["machine"],
        "reproduction": reproduction,
        "quality_uncertainty": _quality_uncertainty(reproduction),
        "cost_telemetry": _paired_costs(rows),
        "prediction_throughput": _prediction(rows),
        "decision": {
            "correctness_clear": True,
            "reproduction_clear": True,
            "public_exposure_stop_condition_present": False,
            "public_exposure_authorized_by_this_result": False,
            "performance_or_cost_gated": False,
        },
        "claims": campaign.claim_spec(),
    }
    return result


def _format_note(result: Mapping[str, Any]) -> str:
    reproduction = result["reproduction"]["aggregate"]
    sports = result["quality_uncertainty"]["sports"]
    general = result["quality_uncertainty"]["general"]
    costs = result["cost_telemetry"]
    prediction = result["prediction_throughput"]["comparisons"]
    lines = [
        "# v0.11 private ensemble evidence result",
        "",
        "This is Tier-E descriptive evidence for the private eight-member release candidate. It is not public exposure, M2/M4, a default change, or release authority.",
        "",
        "## Reproduction and quality",
        "",
        f"- Immutable M3b reproduction: **PASS** at absolute ratio tolerance `{campaign.REPRODUCTION_ABS_TOLERANCE:.1e}`.",
        f"- Pooled v3/single primary-loss ratio: `{reproduction['pooled']['current_ratio']['median']:.6f}`.",
        f"- Sports ratio: `{sports['point_ratio']:.6f}`; season-cluster bootstrap 95% descriptive interval `[{sports['bootstrap_percentiles']['p2_5']:.6f}, {sports['bootstrap_percentiles']['p97_5']:.6f}]`.",
        f"- General ratio: `{general['point_ratio']:.6f}`; case-bootstrap 95% descriptive interval `[{general['bootstrap_percentiles']['p2_5']:.6f}, {general['bootstrap_percentiles']['p97_5']:.6f}]`.",
        "- The sports unit is three season clusters; the four general cells are fixed seeded cases. The 13 cells are not presented as independent datasets.",
        "",
        "## Cost telemetry",
        "",
        "| Comparison | Fit | Peak RSS | RSS delta | Archive bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("v3_vs_single", "v3_vs_existing_bootstrap8"):
        values = costs[name]["equal_case_geomean_ratio"]
        lines.append(
            f"| {name} | {values['fit_seconds']:.3f}x | {values['peak_rss_bytes']:.3f}x | {values['peak_rss_delta_bytes']:.3f}x | {values['archive_bytes']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "Ratios are telemetry beside absolute per-case values in the JSON result. The historical archive-size gate remains retracted.",
            "",
            "## Prediction throughput",
            "",
            "Seconds ratios below one favor the numerator. Each aggregate is the equal-coordinate geometric mean of three-block median paired ratios.",
            "",
            "| Comparison | Seconds ratio | Faster / 16 coordinates |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, values in prediction.items():
        lines.append(
            f"| {name} | {values['equal_coordinate_geomean_seconds_ratio']:.3f}x | {values['coordinates_numerator_faster']} |"
        )
    lines.extend(
        [
            "",
            "## Disposition",
            "",
            "No correctness or reproduction stop condition is present. This result does not itself authorize public exposure; all performance, cost, and dispersion findings are disclosures rather than gates.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_create_only(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"analysis output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--contract", type=Path, default=campaign.CONTRACT_PATH)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args(argv)
    for name in ("raw", "contract", "result", "note"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.raw, args.contract)
    _write_create_only(args.result, campaign.json_bytes(result))
    _write_create_only(args.note, _format_note(result).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
