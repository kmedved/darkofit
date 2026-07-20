#!/usr/bin/env python3
"""Analyze the frozen Wave-1 M3a phase artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "benchmarks" / "run_m3a_wave1.py"
CONTRACT_PATH = ROOT / "benchmarks" / "m3a_wave1_contract.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "m3a_wave1.json"
DEFAULT_REPORT = ROOT / "benchmarks" / "m3a_wave1_result.md"
PRIMARY_DECISION_PREFIX = "M3A_PRIMARY_DECISION="


def _load_runner():
    name = "m3a_frozen_runner_for_analysis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen M3a runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise RuntimeError("M3a geometric-mean inputs must be positive")
    return float(np.exp(np.mean(np.log(array))))


def season_cluster_summary(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("M3a season-cluster summary has no rows")
    seasons = sorted({int(row["season"]) for row in rows})
    targets_by_season = {
        season: [row for row in rows if int(row["season"]) == season]
        for season in seasons
    }
    if len(seasons) != 3 or any(
        len(season_rows) != 3 for season_rows in targets_by_season.values()
    ):
        raise RuntimeError("M3a cluster summary requires three targets x seasons")
    season_logs = np.asarray(
        [
            np.mean(
                np.log(
                    np.asarray(
                        [row["ratio"] for row in targets_by_season[season]],
                        dtype=np.float64,
                    )
                )
            )
            for season in seasons
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(
        0, len(seasons), size=(int(resamples), len(seasons))
    )
    bootstrap = np.exp(np.mean(season_logs[sampled], axis=1))
    return {
        "cell_geometric_mean": _geomean(row["ratio"] for row in rows),
        "cell_min": float(min(row["ratio"] for row in rows)),
        "cell_max": float(max(row["ratio"] for row in rows)),
        "season_ratios": {
            str(season): float(math.exp(season_logs[index]))
            for index, season in enumerate(seasons)
        },
        "cluster_bootstrap": {
            "unit": "season",
            "clusters": len(seasons),
            "seed": int(seed),
            "resamples": int(resamples),
            "p2_5": float(np.quantile(bootstrap, 0.025)),
            "p50": float(np.quantile(bootstrap, 0.50)),
            "p95": float(np.quantile(bootstrap, 0.95)),
            "p97_5": float(np.quantile(bootstrap, 0.975)),
            "descriptive_only": True,
        },
    }


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_arms(
    phase: str,
    contract: dict[str, Any],
) -> tuple[str, ...]:
    orders = contract["execution"]["orders"][phase]
    return tuple(arm for order in orders for arm in order)


def _validate_phase(
    artifact: dict[str, Any],
    *,
    phase: str,
    contract: dict[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    runner = _load_runner()
    structural = (
        artifact.get("schema_version") == 1
        and artifact.get("name") == "wave1_m3a_phase"
        and artifact.get("phase") == phase
        and artifact.get("contract", {}).get("sha256")
        == _sha256(contract_path)
        and artifact.get("contract", {}).get("runner_sha256")
        == contract["bound_files"]["runner"]["sha256"]
        and artifact.get("contract", {}).get("analyzer_sha256")
        == contract["bound_files"]["analyzer"]["sha256"]
    )
    if not structural:
        raise RuntimeError(f"M3a {phase} artifact violates its frozen contract")
    expected = _expected_arms(phase, contract)
    observed = tuple(row["arm"] for row in artifact["results"])
    if observed != expected:
        raise RuntimeError(
            f"M3a {phase} arm order changed: {observed!r} != {expected!r}"
        )
    if artifact["orders"] != contract["execution"]["orders"][phase]:
        raise RuntimeError(f"M3a {phase} block orders changed")
    sources = artifact["sources"]
    source_pins_valid = (
        sources["darkofit"]["head"] == runner.DARKO_SOURCE_HEAD
        and sources["chimeraboost"]["head"] == runner.CHIMERA_SOURCE_HEAD
        and all(state["clean"] for state in sources.values())
    )
    no_worker_stderr = all(
        row.get("worker_stderr") is None for row in artifact["results"]
    )
    rss_valid = all(
        row["rss_sampling"]["samples"] >= 2
        and not row["rss_sampling"]["errors"]
        and row["aggregate_peak_rss_bytes"] > 0
        for row in artifact["results"]
    )
    expected_general = set(contract["general"]["arms"])
    general_grid_valid = True
    for row in artifact["results"]:
        expected_cells = (
            len(contract["general"]["datasets"])
            * len(contract["general"]["seeds"])
            if row["arm"] in expected_general
            else 0
        )
        if len(row["general_cells"]) != expected_cells:
            general_grid_valid = False
    sports_grid_valid = all(
        len(row["sports_cells"]) == 9 for row in artifact["results"]
    )
    return {
        "structural_contract": True,
        "source_pins_valid": source_pins_valid,
        "no_worker_stderr": no_worker_stderr,
        "rss_valid": rss_valid,
        "sports_grid_valid": sports_grid_valid,
        "general_grid_valid": general_grid_valid,
        "passed": all(
            (
                source_pins_valid,
                no_worker_stderr,
                rss_valid,
                sports_grid_valid,
                general_grid_valid,
            )
        ),
    }


def _single_results(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in artifact["results"]:
        arm = row["arm"]
        if arm in results:
            raise RuntimeError(f"M3a expected one {arm} result in this phase")
        results[arm] = row
    return results


def _cell_map(result: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    cells = {
        (int(row["season"]), str(row["target"])): row
        for row in result["sports_cells"]
    }
    if len(cells) != 9:
        raise RuntimeError(f"M3a arm {result['arm']} has duplicate sports cells")
    return cells


def _ratio_rows(
    control: dict[str, Any],
    candidate: dict[str, Any],
    view: str,
) -> list[dict[str, Any]]:
    controls = _cell_map(control)
    candidates = _cell_map(candidate)
    if controls.keys() != candidates.keys():
        raise RuntimeError("M3a paired arms have different sports cells")
    rows = []
    for season, target in sorted(controls):
        left = controls[(season, target)][view]
        right = candidates[(season, target)][view]
        if left["target_sha256"] != right["target_sha256"]:
            raise RuntimeError("M3a paired sports targets differ")
        if left["rmse"] <= 0.0 or right["rmse"] <= 0.0:
            raise RuntimeError("M3a paired sports RMSE is non-positive")
        rows.append(
            {
                "season": season,
                "target": target,
                "control_rmse": float(left["rmse"]),
                "candidate_rmse": float(right["rmse"]),
                "ratio": float(right["rmse"] / left["rmse"]),
            }
        )
    return rows


def sports_pair_summary(
    control: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    views = {}
    for view in (
        "player_disjoint",
        "creator",
        "held_team",
        "seen_player",
        "cold_player",
    ):
        rows = _ratio_rows(control, candidate, view)
        views[view] = {
            "rows": rows,
            **season_cluster_summary(
                rows,
                seed=contract["inference"]["seed"],
                resamples=contract["inference"]["resamples"],
            ),
        }
    return {
        "control": control["arm"],
        "candidate": candidate["arm"],
        "views": views,
    }


def _cost_ratio(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = control["costs"]["player_plus_held"]
    right = candidate["costs"]["player_plus_held"]

    def ratio(numerator: float, denominator: float) -> float:
        if numerator < 0.0 or denominator <= 0.0:
            raise RuntimeError("M3a cost ratio has an invalid denominator")
        return float(numerator / denominator)

    return {
        "fit_seconds": {
            "control": float(left["fit_seconds"]),
            "candidate": float(right["fit_seconds"]),
            "ratio": ratio(right["fit_seconds"], left["fit_seconds"]),
        },
        "predict_seconds": {
            "control": float(left["predict_seconds"]),
            "candidate": float(right["predict_seconds"]),
            "ratio": ratio(
                right["predict_seconds"], left["predict_seconds"]
            ),
        },
        "held_median_model_bytes": {
            "control": float(left["held_median_model_bytes"]),
            "candidate": float(right["held_median_model_bytes"]),
            "ratio": ratio(
                right["held_median_model_bytes"],
                left["held_median_model_bytes"],
            ),
        },
        "aggregate_peak_rss_bytes": {
            "control": int(control["aggregate_peak_rss_bytes"]),
            "candidate": int(candidate["aggregate_peak_rss_bytes"]),
            "ratio": ratio(
                candidate["aggregate_peak_rss_bytes"],
                control["aggregate_peak_rss_bytes"],
            ),
        },
        "single_pass_descriptive": True,
    }


def _pair_target_integrity(
    results: dict[str, dict[str, Any]],
) -> bool:
    arms = sorted(results)
    reference = _cell_map(results[arms[0]])
    for arm in arms[1:]:
        candidate = _cell_map(results[arm])
        if candidate.keys() != reference.keys():
            return False
        for key in reference:
            for view in (
                "player_disjoint",
                "creator",
                "held_team",
                "seen_player",
                "cold_player",
            ):
                if (
                    reference[key][view]["target_sha256"]
                    != candidate[key][view]["target_sha256"]
                ):
                    return False
    return True


def _checks(
    pair: dict[str, Any],
    costs: dict[str, Any],
    contract: dict[str, Any],
    integrity_passed: bool,
) -> dict[str, Any]:
    gates = contract["survival_gates"]
    primary = pair["views"]["player_disjoint"]
    held = pair["views"]["held_team"]
    cold = pair["views"]["cold_player"]
    season_values = list(primary["season_ratios"].values())
    checks = {
        "integrity": {
            "value": bool(integrity_passed),
            "required": True,
            "passed": bool(integrity_passed),
        },
        "player_geomean": {
            "value": primary["cell_geometric_mean"],
            "at_most": gates["player_geomean_at_most"],
        },
        "player_cluster_p95": {
            "value": primary["cluster_bootstrap"]["p95"],
            "at_most": gates["player_cluster_p95_at_most"],
        },
        "held_geomean": {
            "value": held["cell_geometric_mean"],
            "at_most": gates["held_geomean_at_most"],
        },
        "cold_geomean": {
            "value": cold["cell_geometric_mean"],
            "at_most": gates["cold_geomean_at_most"],
        },
        "worst_season": {
            "value": max(season_values),
            "at_most": gates["worst_season_at_most"],
        },
        "worst_player_cell": {
            "value": primary["cell_max"],
            "at_most": gates["worst_player_cell_at_most"],
        },
        "fit_cost": {
            "value": costs["fit_seconds"]["ratio"],
            "at_most": gates["fit_ratio_at_most"],
        },
        "predict_cost": {
            "value": costs["predict_seconds"]["ratio"],
            "at_most": gates["predict_ratio_at_most"],
        },
        "model_bytes": {
            "value": costs["held_median_model_bytes"]["ratio"],
            "at_most": gates["model_bytes_ratio_at_most"],
        },
        "peak_rss": {
            "value": costs["aggregate_peak_rss_bytes"]["ratio"],
            "at_most": gates["peak_rss_ratio_at_most"],
        },
    }
    for name, record in checks.items():
        if name == "integrity":
            continue
        record["passed"] = bool(record["value"] <= record["at_most"])
    return checks


def primary_decision(
    primary_artifact: Path,
    *,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    runner = _load_runner()
    contract = runner.load_contract(contract_path)
    artifact = _load_artifact(primary_artifact)
    integrity = _validate_phase(
        artifact,
        phase="primary-quality",
        contract=contract,
        contract_path=contract_path,
    )
    results = _single_results(artifact)
    target_integrity = _pair_target_integrity(results)
    integrity["paired_targets_identical"] = target_integrity
    integrity["passed"] = bool(integrity["passed"] and target_integrity)
    pair = sports_pair_summary(
        results[runner.DARKO_SINGLE],
        results[runner.DARKO_GROUP8],
        contract,
    )
    costs = _cost_ratio(
        results[runner.DARKO_SINGLE],
        results[runner.DARKO_GROUP8],
    )
    checks = _checks(pair, costs, contract, integrity["passed"])
    survives = all(record["passed"] for record in checks.values())
    return {
        "integrity": integrity,
        "darkofit_group8_vs_single": pair,
        "costs": costs,
        "checks": checks,
        "survives": bool(survives),
        "timing_repeats_authorized": bool(survives),
        "provisional_track_b_disposition": (
            "continue_private_ensemble_v3_program"
            if survives
            else "close_preserve_current_opt_in"
        ),
        "default_change_authorized": False,
    }


def _general_cell_map(
    result: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["dataset"], row["size"], int(row["seed"])): row
        for row in result["general_cells"]
    }


def general_pair_summary(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = _general_cell_map(control)
    right = _general_cell_map(candidate)
    if not left or left.keys() != right.keys():
        raise RuntimeError("M3a paired general grids differ")
    rows = []
    for key in sorted(left):
        left_score = left[key]["score"]
        right_score = right[key]["score"]
        if left_score["target_sha256"] != right_score["target_sha256"]:
            raise RuntimeError("M3a paired general targets differ")
        rows.append(
            {
                "dataset": key[0],
                "size": key[1],
                "seed": key[2],
                "control_rmse": left_score["rmse"],
                "candidate_rmse": right_score["rmse"],
                "ratio": right_score["rmse"] / left_score["rmse"],
            }
        )
    return {
        "control": control["arm"],
        "candidate": candidate["arm"],
        "rows": rows,
        "geometric_mean": _geomean(row["ratio"] for row in rows),
        "wins": sum(row["ratio"] < 1.0 for row in rows),
        "cells": len(rows),
        "descriptive_only": True,
    }


def _primary_timing_summary(
    primary: dict[str, Any],
    repeats: dict[str, Any] | None,
    contract: dict[str, Any],
    survives: bool,
) -> dict[str, Any]:
    runner = _load_runner()
    runs = list(primary["results"])
    if survives:
        if repeats is None:
            raise RuntimeError("surviving M3a quality pass requires repeat artifact")
        _validate_phase(
            repeats,
            phase="primary-repeats",
            contract=contract,
            contract_path=CONTRACT_PATH,
        )
        runs.extend(repeats["results"])
    elif repeats is not None:
        raise RuntimeError("M3a repeat artifact exists after a failed quality gate")
    by_arm = {arm: [] for arm in runner.PRIMARY_ARMS}
    for row in runs:
        by_arm[row["arm"]].append(row)
    expected_count = 3 if survives else 1
    if any(len(values) != expected_count for values in by_arm.values()):
        raise RuntimeError("M3a primary timing series has the wrong repeat count")
    summaries = {}
    for arm, values in by_arm.items():
        fit = [
            row["costs"]["player_plus_held"]["fit_seconds"] for row in values
        ]
        predict = [
            row["costs"]["player_plus_held"]["predict_seconds"]
            for row in values
        ]
        rss = [row["aggregate_peak_rss_bytes"] for row in values]
        model_bytes = [
            row["costs"]["player_plus_held"]["held_median_model_bytes"]
            for row in values
        ]
        summaries[arm] = {
            "observations": len(values),
            "fit_seconds": fit,
            "fit_median": float(median(fit)),
            "predict_seconds": predict,
            "predict_median": float(median(predict)),
            "aggregate_peak_rss_bytes": rss,
            "aggregate_peak_rss_median": float(median(rss)),
            "held_model_bytes": model_bytes,
            "held_model_bytes_median": float(median(model_bytes)),
            "behavior_fingerprints": [
                row["behavior_fingerprint_sha256"] for row in values
            ],
            "behavior_stable": len(
                {row["behavior_fingerprint_sha256"] for row in values}
            )
            == 1,
        }
    return {
        "repeat_series_run": bool(survives),
        "reason": (
            "group-safe quality survived; two additional blocks required"
            if survives
            else "group-safe quality did not survive; repeats forbidden"
        ),
        "arms": summaries,
        "timing_decision_eligible": bool(
            survives
            and all(summary["behavior_stable"] for summary in summaries.values())
        ),
    }


def analyze_campaign(
    primary_path: Path,
    diagnostics_path: Path,
    *,
    repeats_path: Path | None = None,
    contract_path: Path = CONTRACT_PATH,
) -> tuple[dict[str, Any], str]:
    runner = _load_runner()
    contract = runner.load_contract(contract_path)
    primary = _load_artifact(primary_path)
    diagnostics = _load_artifact(diagnostics_path)
    primary_result = primary_decision(
        primary_path, contract_path=contract_path
    )
    diagnostic_integrity = _validate_phase(
        diagnostics,
        phase="diagnostics",
        contract=contract,
        contract_path=contract_path,
    )
    primary_runs = _single_results(primary)
    diagnostic_runs = _single_results(diagnostics)
    repeats = None if repeats_path is None else _load_artifact(repeats_path)

    sports_pairs = {
        "darkofit_group8_vs_single": (
            primary_result["darkofit_group8_vs_single"]
        ),
        "chimeraboost_ensemble8_vs_single": sports_pair_summary(
            primary_runs[runner.CHIMERA_SINGLE],
            primary_runs[runner.CHIMERA_ENSEMBLE8],
            contract,
        ),
        "darkofit_row5_vs_single": sports_pair_summary(
            primary_runs[runner.DARKO_SINGLE],
            diagnostic_runs[runner.DARKO_ROW5],
            contract,
        ),
        "darkofit_row8_vs_single": sports_pair_summary(
            primary_runs[runner.DARKO_SINGLE],
            diagnostic_runs[runner.DARKO_ROW8],
            contract,
        ),
        "darkofit_group5_vs_single": sports_pair_summary(
            primary_runs[runner.DARKO_SINGLE],
            diagnostic_runs[runner.DARKO_GROUP5],
            contract,
        ),
        "chimeraboost_float_single_vs_quantized_single": sports_pair_summary(
            primary_runs[runner.CHIMERA_SINGLE],
            diagnostic_runs[runner.CHIMERA_FLOAT_SINGLE],
            contract,
        ),
        "chimeraboost_float_ensemble8_vs_quantized_ensemble8": (
            sports_pair_summary(
                primary_runs[runner.CHIMERA_ENSEMBLE8],
                diagnostic_runs[runner.CHIMERA_FLOAT_ENSEMBLE8],
                contract,
            )
        ),
        "chimeraboost_float_ensemble8_vs_float_single": sports_pair_summary(
            diagnostic_runs[runner.CHIMERA_FLOAT_SINGLE],
            diagnostic_runs[runner.CHIMERA_FLOAT_ENSEMBLE8],
            contract,
        ),
    }
    general_pairs = {
        "darkofit_row8_vs_single": general_pair_summary(
            primary_runs[runner.DARKO_SINGLE],
            diagnostic_runs[runner.DARKO_ROW8],
        ),
        "chimeraboost_ensemble8_vs_single": general_pair_summary(
            primary_runs[runner.CHIMERA_SINGLE],
            primary_runs[runner.CHIMERA_ENSEMBLE8],
        ),
    }
    timing = _primary_timing_summary(
        primary,
        repeats,
        contract,
        primary_result["survives"],
    )
    all_integrity = bool(
        primary_result["integrity"]["passed"]
        and diagnostic_integrity["passed"]
        and (
            not primary_result["survives"]
            or timing["timing_decision_eligible"]
        )
    )
    disposition = (
        "continue_private_ensemble_v3_program"
        if primary_result["survives"] and all_integrity
        else "close_preserve_current_opt_in"
    )
    shard_paths = {
        "primary_quality": primary_path,
        "diagnostics": diagnostics_path,
    }
    if repeats_path is not None:
        shard_paths["primary_repeats"] = repeats_path
    artifact = {
        "schema_version": 1,
        "name": "wave1_m3a_shipped_ensemble_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
            "protocol_sha256": contract["bound_files"]["protocol"]["sha256"],
            "runner_sha256": contract["bound_files"]["runner"]["sha256"],
            "analyzer_sha256": contract["bound_files"]["analyzer"]["sha256"],
        },
        "shards": {
            name: {
                "source_path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in shard_paths.items()
        },
        "raw_phases": {
            "primary_quality": primary,
            "diagnostics": diagnostics,
            "primary_repeats": repeats,
        },
        "analysis": {
            "primary_decision": primary_result,
            "diagnostic_integrity": diagnostic_integrity,
            "sports_pairs": sports_pairs,
            "general_pairs": general_pairs,
            "primary_timing": timing,
            "all_required_integrity": all_integrity,
            "track_b_disposition": disposition,
            "default_change_authorized": False,
            "cross_season_generalization_authorized": False,
            "m6_ranking_authorized": False,
        },
    }
    report = render_report(artifact)
    return artifact, report


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def render_report(artifact: dict[str, Any]) -> str:
    analysis = artifact["analysis"]
    primary = analysis["primary_decision"]
    group = primary["darkofit_group8_vs_single"]["views"]
    chimera = analysis["sports_pairs"][
        "chimeraboost_ensemble8_vs_single"
    ]["views"]
    general = analysis["general_pairs"]
    timing = analysis["primary_timing"]
    lines = [
        "# Wave 1 M3a result",
        "",
        "## Outcome",
        "",
        (
            f"Track B disposition: **{analysis['track_b_disposition']}**. "
            f"The frozen group-safe survival rule "
            f"{'passed' if primary['survives'] else 'did not pass'}."
        ),
        "",
        "No default change or cross-season generalization is authorized.",
        "",
        "## Player-disjoint quality",
        "",
        "| Pair | Primary geomean | Season-cluster p95 | Held-team | Cold-player |",
        "|---|---:|---:|---:|---:|",
        (
            "| DarkoFit group8 / single | "
            f"{_fmt(group['player_disjoint']['cell_geometric_mean'])} | "
            f"{_fmt(group['player_disjoint']['cluster_bootstrap']['p95'])} | "
            f"{_fmt(group['held_team']['cell_geometric_mean'])} | "
            f"{_fmt(group['cold_player']['cell_geometric_mean'])} |"
        ),
        (
            "| ChimeraBoost ensemble8 / single | "
            f"{_fmt(chimera['player_disjoint']['cell_geometric_mean'])} | "
            f"{_fmt(chimera['player_disjoint']['cluster_bootstrap']['p95'])} | "
            f"{_fmt(chimera['held_team']['cell_geometric_mean'])} | "
            f"{_fmt(chimera['cold_player']['cell_geometric_mean'])} |"
        ),
        "",
        "The interval resamples only three spent season clusters and is "
        "descriptive.",
        "",
        "## Frozen DarkoFit survival checks",
        "",
        "| Check | Value | Limit | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for name, record in primary["checks"].items():
        if name == "integrity":
            lines.append(
                f"| {name} | {record['value']} | required | "
                f"{'yes' if record['passed'] else 'no'} |"
            )
        else:
            lines.append(
                f"| {name} | {_fmt(record['value'])} | "
                f"{_fmt(record['at_most'])} | "
                f"{'yes' if record['passed'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## General medium-slice context",
            "",
            "| Pair | Geomean RMSE ratio | Wins | Cells |",
            "|---|---:|---:|---:|",
            (
                "| DarkoFit row8 / single | "
                f"{_fmt(general['darkofit_row8_vs_single']['geometric_mean'])} | "
                f"{general['darkofit_row8_vs_single']['wins']} | "
                f"{general['darkofit_row8_vs_single']['cells']} |"
            ),
            (
                "| ChimeraBoost ensemble8 / single | "
                f"{_fmt(general['chimeraboost_ensemble8_vs_single']['geometric_mean'])} | "
                f"{general['chimeraboost_ensemble8_vs_single']['wins']} | "
                f"{general['chimeraboost_ensemble8_vs_single']['cells']} |"
            ),
            "",
            "These row-sampling cells are descriptive and cannot rescue a "
            "failed group-safe sports result.",
            "",
            "## Timing handling",
            "",
            (
                f"Primary repeat series run: **"
                f"{'yes' if timing['repeat_series_run'] else 'no'}** — "
                f"{timing['reason']}."
            ),
            "",
            "Diagnostic costs are single warmed observations. Aggregate RSS "
            "is a sampled worker-plus-child-process value.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--repeats", type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--primary-decision", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.primary_decision:
        decision = primary_decision(
            args.primary, contract_path=args.contract
        )
        print(
            PRIMARY_DECISION_PREFIX
            + json.dumps(decision, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.diagnostics is None:
        raise RuntimeError("full M3a analysis requires --diagnostics")
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing M3a output: {args.output}")
    if args.report.exists() or args.report.is_symlink():
        raise RuntimeError(f"refusing existing M3a report: {args.report}")
    artifact, report = analyze_campaign(
        args.primary,
        args.diagnostics,
        repeats_path=args.repeats,
        contract_path=args.contract,
    )
    artifact_payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    report_payload = report.encode("utf-8")
    _atomic_create(args.output, artifact_payload)
    try:
        _atomic_create(args.report, report_payload)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": hashlib.sha256(artifact_payload).hexdigest(),
                "report": str(args.report),
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "track_b_disposition": artifact["analysis"][
                    "track_b_disposition"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
