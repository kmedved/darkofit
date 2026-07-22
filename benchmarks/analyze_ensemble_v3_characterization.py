#!/usr/bin/env python3
"""Analyze the frozen ensemble-v3 characterization without shipping claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import run_ensemble_v3_characterization as campaign
except ImportError:  # direct script execution
    import run_ensemble_v3_characterization as campaign


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "benchmarks" / "ensemble_v3_characterization_raw.json"
DEFAULT_RESULT = ROOT / "benchmarks" / "ensemble_v3_characterization_result.json"
DEFAULT_NOTE = ROOT / "benchmarks" / "ensemble_v3_characterization_result.md"
HISTORICAL_RESULT = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_result.json"
HISTORICAL_TIMING = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_timing.json"
HISTORICAL_READOUT = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
)
SPORTS_BOOTSTRAP_SEED = 20_260_720
GENERAL_BOOTSTRAP_SEED = 20_260_721
BOOTSTRAP_DRAWS = 100_000


def _gmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise RuntimeError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    result = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "p2_5": float(result[0]),
        "p50": float(result[1]),
        "p97_5": float(result[2]),
    }


def _bootstrap_gmean(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    distribution = np.exp(np.mean(np.log(array[draws]), axis=1))
    return {
        "seed": seed,
        "draws": BOOTSTRAP_DRAWS,
        "percentiles": _percentiles(distribution),
    }


def analyze_quality(readout: Mapping[str, Any]) -> dict[str, Any]:
    combined = readout["arms_vs_single"]["b1_b2_combined"]
    ratios = {str(key): float(value) for key, value in combined["per_case_primary_ratio"].items()}
    if set(ratios) != {
        *(f"sports_{season}_{target}" for season in (2014, 2015, 2016) for target in ("box_plus_minus", "game_score", "minutes_per_game")),
        "general_categorical_multiclass",
        "general_categorical_reg",
        "general_friedman_numeric",
        "general_numeric_binary",
    }:
        raise RuntimeError("historical quality readout case set drifted")
    sports = {name: value for name, value in ratios.items() if name.startswith("sports_")}
    general = {name: value for name, value in ratios.items() if name.startswith("general_")}
    season_ratios = {}
    for season in (2014, 2015, 2016):
        season_ratios[str(season)] = _gmean(
            [value for name, value in sports.items() if name.startswith(f"sports_{season}_")]
        )
    season_values = list(season_ratios.values())
    sports_loo = {
        omitted: _gmean([value for season, value in season_ratios.items() if season != omitted])
        for omitted in season_ratios
    }
    general_loo = {
        omitted: _gmean([value for case, value in general.items() if case != omitted])
        for omitted in general
    }
    calculated = {
        "all_case_geometric_mean": _gmean(list(ratios.values())),
        "sports_geometric_mean": _gmean(list(sports.values())),
        "general_geometric_mean": _gmean(list(general.values())),
        "wins_vs_single": sum(value < 1.0 for value in ratios.values()),
    }
    for name, expected in (
        ("all_case_geometric_mean", combined["all_case_geometric_mean"]),
        ("sports_geometric_mean", combined["sports_geometric_mean"]),
        ("general_geometric_mean", combined["general_geometric_mean"]),
    ):
        if not math.isclose(calculated[name], float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"historical quality point estimate drifted: {name}")
    return {
        **calculated,
        "case_count": len(ratios),
        "per_case_ratio": ratios,
        "sports": {
            "scope": "player-disjoint cold-player rows within held-team views",
            "season_cluster_count": 3,
            "season_ratios": season_ratios,
            "cluster_bootstrap": _bootstrap_gmean(season_values, seed=SPORTS_BOOTSTRAP_SEED),
            "leave_one_season_out": sports_loo,
            "interpretation": "descriptive dispersion over three spent season clusters",
        },
        "general": {
            "scope": "four fixed medium seeded-75/25 development cases",
            "case_count": 4,
            "bootstrap": _bootstrap_gmean(list(general.values()), seed=GENERAL_BOOTSTRAP_SEED),
            "log_ratio_sample_standard_deviation": float(np.std(np.log(list(general.values())), ddof=1)),
            "leave_one_case_out": general_loo,
            "interpretation": "descriptive dispersion, not four population-independent datasets",
        },
    }


def _quartile_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != campaign.BLOCKS or np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise RuntimeError("paired series is incomplete or invalid")
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    return {
        "values": [float(value) for value in array],
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr_over_median": float((q3 - q1) / median),
    }


def validate_raw(raw: Mapping[str, Any], contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        raw.get("contract_id") != campaign.CONTRACT_ID
        or raw.get("status") != "complete"
        or raw.get("execution") != campaign.execution_spec()
        or raw.get("shipping_or_default_claim_authorized") is not False
        or raw.get("m2_or_m4") is not False
        or raw.get("fresh_or_lockbox_data_used") is not False
        or raw.get("contract", {}).get("sha256") != campaign.sha256(campaign.CONTRACT_PATH)
        or raw.get("contract", {}).get("bindings") != contract["bindings"]
    ):
        raise RuntimeError("raw characterization header is invalid")
    sources = raw.get("sources", {})
    if (
        sources.get("darkofit", {}).get("head") != campaign.DARKOFIT_HEAD
        or sources.get("chimeraboost", {}).get("head") != campaign.CHIMERABOOST_HEAD
        or any(not sources.get(name, {}).get("clean") for name in ("harness", "darkofit", "chimeraboost"))
    ):
        raise RuntimeError("raw characterization source binding is invalid")
    rows = raw.get("rows")
    expected = {
        (block, case_id, arm)
        for block in range(campaign.BLOCKS)
        for case_id in campaign.CASES
        for arm in campaign.ARMS
    }
    observed = set()
    if not isinstance(rows, list):
        raise RuntimeError("raw characterization rows are missing")
    for row in rows:
        key = (row.get("block"), row.get("case_id"), row.get("arm"))
        if key in observed:
            raise RuntimeError(f"duplicate raw characterization row: {key}")
        observed.add(key)
        predictions = row.get("predictions", {})
        if set(predictions) != {str(value) for value in campaign.BATCH_SIZES}:
            raise RuntimeError(f"prediction grid is incomplete: {key}")
        if row.get("fit_rss", {}).get("scope") != "worker_plus_recursive_children":
            raise RuntimeError(f"RSS scope drifted: {key}")
        if row.get("fit_rss", {}).get("errors"):
            raise RuntimeError(f"RSS sampling recorded an error: {key}")
        if row.get("arm") in {campaign.DARKO_SINGLE, campaign.DARKO_V3} and (
            row.get("archive", {}).get("format") != "darkofit_safe_npz"
            or row.get("archive", {}).get("roundtrip_exact") is not True
        ):
            raise RuntimeError(f"DarkoFit archive invariant failed: {key}")
        for batch_size, timing in predictions.items():
            calls = timing.get("calls")
            interval = timing.get("interval_seconds")
            seconds = timing.get("seconds_per_call")
            throughput = timing.get("rows_per_second")
            if (
                isinstance(calls, bool)
                or not isinstance(calls, int)
                or not campaign.MIN_CALLS <= calls <= campaign.MAX_CALLS
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0
                    for value in (interval, seconds, throughput)
                )
                or timing.get("minimum_interval_met")
                != (float(interval) >= campaign.MIN_INTERVAL_SECONDS)
                or timing.get("method") != "predict"
                or timing.get("rows") != int(batch_size)
            ):
                raise RuntimeError(f"prediction timing is invalid: {key}/{batch_size}")
    if observed != expected:
        raise RuntimeError("raw characterization does not contain the exact grid")
    for case_id in campaign.CASES:
        for batch_size in map(str, campaign.BATCH_SIZES):
            inputs = {
                row["predictions"][batch_size]["input_sha256"]
                for row in rows
                if row["case_id"] == case_id
            }
            if len(inputs) != 1:
                raise RuntimeError(f"prediction input drifted: {case_id}/{batch_size}")
            for arm in campaign.ARMS:
                outputs = {
                    row["predictions"][batch_size]["prediction_sha256"]
                    for row in rows
                    if row["case_id"] == case_id and row["arm"] == arm
                }
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"prediction behavior drifted: {case_id}/{batch_size}/{arm}"
                    )
    return rows


def _row_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = (int(row["block"]), str(row["case_id"]), str(row["arm"]))
        if key in result:
            raise RuntimeError(f"duplicate raw characterization row: {key}")
        result[key] = row
    expected = {
        (block, case_id, arm)
        for block in range(campaign.BLOCKS)
        for case_id in campaign.CASES
        for arm in campaign.ARMS
    }
    if set(result) != expected:
        raise RuntimeError("raw characterization does not contain the exact grid")
    return result


def _resource_metric(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    extractor,
) -> dict[str, Any]:
    cases = {}
    for case_id in campaign.CASES:
        ratios = []
        numerator = []
        denominator = []
        for block in range(campaign.BLOCKS):
            single = float(extractor(index[(block, case_id, campaign.DARKO_SINGLE)]))
            v3 = float(extractor(index[(block, case_id, campaign.DARKO_V3)]))
            if single <= 0 or v3 <= 0:
                raise RuntimeError(f"resource metric is nonpositive: {case_id}")
            denominator.append(single)
            numerator.append(v3)
            ratios.append(v3 / single)
        cases[case_id] = {
            "v3_values": numerator,
            "single_values": denominator,
            "v3_over_single": _quartile_summary(ratios),
        }
    medians = [row["v3_over_single"]["median"] for row in cases.values()]
    return {
        "per_case": cases,
        "equal_case_geometric_mean_ratio": _gmean(medians),
        "median_of_case_median_ratios": float(np.median(medians)),
    }


def analyze_resources(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = _row_index(rows)
    fit = _resource_metric(index, lambda row: row["fit_seconds"])
    peak = _resource_metric(index, lambda row: row["fit_rss"]["peak_bytes"])
    archive = _resource_metric(index, lambda row: row["archive"]["bytes"])
    deltas = {}
    for case_id in campaign.CASES:
        v3_values = [
            int(index[(block, case_id, campaign.DARKO_V3)]["fit_rss"]["peak_delta_bytes"])
            for block in range(campaign.BLOCKS)
        ]
        single_values = [
            int(index[(block, case_id, campaign.DARKO_SINGLE)]["fit_rss"]["peak_delta_bytes"])
            for block in range(campaign.BLOCKS)
        ]
        paired_ratios = [
            (None if single <= 0 else float(v3 / single))
            for v3, single in zip(v3_values, single_values)
        ]
        deltas[case_id] = {
            "v3_peak_delta_bytes": v3_values,
            "single_peak_delta_bytes": single_values,
            "paired_ratios": paired_ratios,
            "v3_median_peak_delta_bytes": float(np.median(v3_values)),
            "single_median_peak_delta_bytes": float(np.median(single_values)),
        }
    return {
        "scope": "four frozen medium general cases; three fresh blocks",
        "fit_seconds": fit,
        "process_tree_peak_rss_bytes": peak,
        "process_tree_peak_minus_start_bytes": deltas,
        "safe_npz_archive_bytes": archive,
    }


def _prediction_pair(
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    case_id: str,
    batch_size: int,
    numerator_arm: str,
    denominator_arm: str,
) -> dict[str, Any]:
    ratios = []
    numerator_throughput = []
    denominator_throughput = []
    key = str(batch_size)
    for block in range(campaign.BLOCKS):
        numerator = index[(block, case_id, numerator_arm)]["predictions"][key]
        denominator = index[(block, case_id, denominator_arm)]["predictions"][key]
        ratios.append(float(numerator["seconds_per_call"]) / float(denominator["seconds_per_call"]))
        numerator_throughput.append(float(numerator["rows_per_second"]))
        denominator_throughput.append(float(denominator["rows_per_second"]))
    return {
        "seconds_ratio": _quartile_summary(ratios),
        "numerator_rows_per_second_median": float(np.median(numerator_throughput)),
        "denominator_rows_per_second_median": float(np.median(denominator_throughput)),
    }


def analyze_prediction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = _row_index(rows)
    pairs = {
        "darkofit_single_over_chimeraboost_0_18_single": (campaign.DARKO_SINGLE, campaign.CHIMERA_SINGLE),
        "darkofit_ensemble_v3_over_chimeraboost_0_18_single": (campaign.DARKO_V3, campaign.CHIMERA_SINGLE),
        "darkofit_ensemble_v3_over_darkofit_single": (campaign.DARKO_V3, campaign.DARKO_SINGLE),
    }
    coordinates = {}
    aggregate = {name: [] for name in pairs}
    short = []
    for case_id in campaign.CASES:
        coordinates[case_id] = {}
        for batch_size in campaign.BATCH_SIZES:
            name = str(batch_size)
            coordinate = {}
            for pair_name, (numerator, denominator) in pairs.items():
                summary = _prediction_pair(index, case_id, batch_size, numerator, denominator)
                coordinate[pair_name] = summary
                aggregate[pair_name].append(summary["seconds_ratio"]["median"])
            coordinates[case_id][name] = coordinate
            for block in range(campaign.BLOCKS):
                for arm in campaign.ARMS:
                    timing = index[(block, case_id, arm)]["predictions"][name]
                    if timing["minimum_interval_met"] is not True:
                        short.append(
                            {
                                "block": block,
                                "case_id": case_id,
                                "arm": arm,
                                "batch_size": batch_size,
                                "interval_seconds": timing["interval_seconds"],
                                "calls": timing["calls"],
                            }
                        )
    aggregates = {}
    for name, values in aggregate.items():
        aggregates[name] = {
            "coordinate_count": len(values),
            "equal_coordinate_geometric_mean_ratio": _gmean(values),
            "coordinates_at_or_below_parity": sum(value <= 1.0 for value in values),
            "worst_coordinate_ratio": max(values),
            "best_coordinate_ratio": min(values),
        }
    return {
        "scope": "four fixed tasks by four batch sizes; complete public predict",
        "minimum_interval_seconds": campaign.MIN_INTERVAL_SECONDS,
        "short_intervals": short,
        "all_intervals_meet_minimum": not short,
        "coordinates": coordinates,
        "aggregate": aggregates,
    }


def _historical_metric(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    index = {(int(row["repeat"]), row["case_id"], row["arm"]): row for row in rows}
    case_ids = sorted({row["case_id"] for row in rows if row["arm"] == "b1_b2_combined"})
    cases = {}
    for case_id in case_ids:
        repeats = sorted({int(row["repeat"]) for row in rows if row["case_id"] == case_id})
        ratios = [
            float(index[(repeat, case_id, "b1_b2_combined")][field])
            / float(index[(repeat, case_id, "single_reference")][field])
            for repeat in repeats
        ]
        cases[case_id] = {"ratios": ratios, "median_ratio": float(np.median(ratios))}
    medians = [value["median_ratio"] for value in cases.values()]
    return {"per_case": cases, "equal_case_geometric_mean_ratio": _gmean(medians)}


def analyze_historical_resources(
    historical_result: Mapping[str, Any], historical_timing: Mapping[str, Any]
) -> dict[str, Any]:
    rows = historical_timing["rows"]
    return {
        "scope": "immutable M3b r3 13-case timing; self-worker RSS, not process-tree",
        "rss_scope": historical_timing["rss_scope"],
        "fit_seconds_v3_over_single": _historical_metric(rows, "fit_seconds"),
        "predict_seconds_v3_over_single": _historical_metric(rows, "predict_seconds"),
        "rss_v3_over_single": _historical_metric(rows, "peak_rss_bytes"),
        "archive_v3_over_single": _historical_metric(rows, "archive_bytes"),
        "stored_candidate_summary": historical_result["candidates"]["b1_b2_combined"]["resources"],
    }


def _fmt_ratio(value: float) -> str:
    return f"{value:.3f}x"


def render_note(result: Mapping[str, Any], result_sha256: str) -> str:
    quality = result["quality"]
    resources = result["current_resources"]
    prediction = result["prediction"]
    sports_ci = quality["sports"]["cluster_bootstrap"]["percentiles"]
    general_ci = quality["general"]["bootstrap"]["percentiles"]
    lines = [
        "# Ensemble-v3 release-candidate characterization",
        "",
        "## Scope",
        "",
        "Tier-E characterization on spent/frozen evidence and one pinned current-source",
        "performance grid. This is not M2, M4, a shipping certificate, a public API",
        "change, or authority to release v0.11.",
        "",
        "## Quality",
        "",
        f"The historical combined recipe beat single on {quality['wins_vs_single']}/{quality['case_count']} cases. ",
        f"Equal-case ratios were {_fmt_ratio(quality['all_case_geometric_mean'])} overall, ",
        f"{_fmt_ratio(quality['sports_geometric_mean'])} on the player-disjoint cold-player sports view, ",
        f"and {_fmt_ratio(quality['general_geometric_mean'])} on the four fixed general cases.",
        "",
        f"The three-season clustered descriptive interval was {_fmt_ratio(sports_ci['p2_5'])}--{_fmt_ratio(sports_ci['p97_5'])}; ",
        f"the four-case general descriptive interval was {_fmt_ratio(general_ci['p2_5'])}--{_fmt_ratio(general_ci['p97_5'])}. ",
        "With only three seasons and four fixed general cases, neither interval is a population-generalization claim.",
        "",
        "## Current fit, memory, and archive telemetry",
        "",
        "| Metric | Ensemble-v3 / DarkoFit single |",
        "| --- | ---: |",
        f"| Fit wall time, equal-case geomean | {_fmt_ratio(resources['fit_seconds']['equal_case_geometric_mean_ratio'])} |",
        f"| Process-tree peak RSS, equal-case geomean | {_fmt_ratio(resources['process_tree_peak_rss_bytes']['equal_case_geometric_mean_ratio'])} |",
        f"| Safe-NPZ bytes, equal-case geomean | {_fmt_ratio(resources['safe_npz_archive_bytes']['equal_case_geometric_mean_ratio'])} |",
        "",
        "RSS is worker-plus-recursive-child peak during formal fit. Absolute peak-minus-start deltas are retained in the JSON; the ratio is not used as a gate.",
        "",
        "## Prediction throughput",
        "",
        "| Comparison | Equal-coordinate time ratio | At/below parity | Worst coordinate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("darkofit_single_over_chimeraboost_0_18_single", "DarkoFit single / ChimeraBoost 0.18 single"),
        ("darkofit_ensemble_v3_over_chimeraboost_0_18_single", "DarkoFit ensemble-v3 / ChimeraBoost 0.18 single"),
        ("darkofit_ensemble_v3_over_darkofit_single", "DarkoFit ensemble-v3 / DarkoFit single"),
    ):
        row = prediction["aggregate"][name]
        lines.append(
            f"| {label} | {_fmt_ratio(row['equal_coordinate_geometric_mean_ratio'])} | "
            f"{row['coordinates_at_or_below_parity']}/{row['coordinate_count']} | "
            f"{_fmt_ratio(row['worst_coordinate_ratio'])} |"
        )
    lines.extend(
        [
            "",
            f"Minimum integrated interval: {prediction['minimum_interval_seconds']:.2f}s. "
            f"Short intervals: {len(prediction['short_intervals'])}. Raw paired series and IQR/median are retained for every coordinate.",
            "",
            "## Interpretation boundary",
            "",
            "Eight members are the only evaluated recipe, not an optimized default. These measurements may inform the later owner ship decision, but they do not expose the API, change defaults, establish sports safety outside the spent panel, or certify prediction performance.",
            "",
            "## Evidence",
            "",
            f"- Raw artifact SHA-256: `{result['raw_artifact']['sha256']}`.",
            f"- Result JSON SHA-256: `{result_sha256}`.",
            f"- Contract SHA-256: `{result['contract']['sha256']}`.",
            f"- DarkoFit source: `{campaign.DARKOFIT_HEAD}`.",
            f"- ChimeraBoost source: `{campaign.CHIMERABOOST_HEAD}`.",
            "- Fresh/lockbox data: none.",
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


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    contract = campaign.load_contract(args.contract)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    rows = validate_raw(raw, contract)
    readout = json.loads(HISTORICAL_READOUT.read_text(encoding="utf-8"))
    historical_result = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    historical_timing = json.loads(HISTORICAL_TIMING.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "characterization_complete",
        "decision": "report_only_await_separate_public_ship_authorization",
        "quality": analyze_quality(readout),
        "current_resources": analyze_resources(rows),
        "historical_resources": analyze_historical_resources(historical_result, historical_timing),
        "prediction": analyze_prediction(rows),
        "raw_artifact": {"path": str(args.raw), "bytes": args.raw.stat().st_size, "sha256": campaign.sha256(args.raw)},
        "contract": {"path": str(args.contract), "sha256": campaign.sha256(args.contract)},
        "claims": {
            "tier": "E",
            "public_api_or_default_change_authorized": False,
            "release_authorized": False,
            "m2_or_m4": False,
            "fresh_or_lockbox_data_used": False,
            "prediction_certified": False,
        },
    }
    result_payload = campaign.json_bytes(result)
    result_sha = hashlib.sha256(result_payload).hexdigest()
    note_payload = render_note(result, result_sha).encode("utf-8")
    _write_create_only(args.result, result_payload)
    try:
        _write_create_only(args.note, note_payload)
    except BaseException:
        args.result.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=campaign.CONTRACT_PATH)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args(argv)
    for name in ("contract", "raw", "result", "note"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    analyze(parse_args())
