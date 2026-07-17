#!/usr/bin/env python3
"""Validate and score a frozen DarkoFit SynthGen raw ledger artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import run_synthgen_darkofit_ledger as runner


ANALYZER_PATH = Path(__file__).resolve()
FREEZE_PATH = BENCHMARKS_ROOT / "synthgen_df1_freeze.json"
DEFAULT_RAW = REPO_ROOT / ".cache" / "synthgen-darkofit-df1-ledger" / "raw.json"
DEFAULT_JSON = BENCHMARKS_ROOT / "synthgen_darkofit_ledger_result.json"
DEFAULT_MARKDOWN = BENCHMARKS_ROOT / "synthgen_darkofit_ledger_result.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(commit: str, path: Path) -> bytes:
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise RuntimeError("geometric mean requires at least one value")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise RuntimeError("geometric mean values must be positive and finite")
    return float(np.exp(np.mean(np.log(array))))


def pair_summary(
    metrics: dict[tuple[int, int, str], float],
    dataset_ids: list[int],
    arm: str,
    comparator: str,
    seeds: tuple[int, ...] = runner.SPLIT_SEEDS,
) -> dict[str, Any]:
    dataset_ratios = {}
    split_ratios = {}
    for dataset_id in dataset_ids:
        ratios = []
        for seed in seeds:
            candidate = metrics[(dataset_id, seed, arm)]
            control = metrics[(dataset_id, seed, comparator)]
            if control <= 0.0:
                raise RuntimeError("control metric must be positive")
            ratio = candidate / control
            ratios.append(ratio)
            split_ratios[f"{dataset_id}:{seed}"] = ratio
        dataset_ratios[str(dataset_id)] = geometric_mean(ratios)
    values = np.asarray(list(dataset_ratios.values()), dtype=np.float64)
    epsilon = runner.EPSILON
    return {
        "arm": arm,
        "comparator": comparator,
        "n_datasets": len(dataset_ids),
        "aggregate_ratio": geometric_mean(values.tolist()),
        "improvement": 1.0 - geometric_mean(values.tolist()),
        "wins": int(np.count_nonzero(values < 1.0 - epsilon)),
        "losses": int(np.count_nonzero(values > 1.0 + epsilon)),
        "ties": int(np.count_nonzero(np.abs(values - 1.0) <= epsilon)),
        "maximum_dataset_regression": float(np.max(values - 1.0)),
        "maximum_split_regression": float(
            max(split_ratios.values()) - 1.0
        ),
        "dataset_ratios": dataset_ratios,
        "split_ratios": split_ratios,
    }


def _record_map(
    records: list[dict[str, Any]],
    *,
    kind: str,
    metric: str,
    expected_ids: list[int],
    expected_seeds: tuple[int, ...],
    expected_arms: tuple[str, ...],
) -> dict[tuple[int, int, str], float]:
    expected = {
        (dataset_id, seed, arm)
        for dataset_id in expected_ids
        for seed in expected_seeds
        for arm in expected_arms
    }
    found = {}
    for record in records:
        if record.get("kind") != kind:
            continue
        coordinate = (
            int(record["dataset_id"]),
            int(record["seed"]),
            str(record["arm"]),
        )
        if coordinate in found:
            raise RuntimeError(f"duplicate raw coordinate {coordinate}")
        value = float(record[metric])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"invalid metric at {coordinate}")
        found[coordinate] = value
    missing = expected - set(found)
    unexpected = set(found) - expected
    if missing or unexpected:
        raise RuntimeError(
            f"raw coordinate boundary failed: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return found


def _validate_dataset_boundary(
    raw: dict[str, Any], manifest: dict[str, Any]
) -> None:
    datasets = raw.get("datasets")
    if not isinstance(datasets, list):
        raise RuntimeError("raw artifact lacks dataset records")
    expected = {
        ("regression_ledger", int(dataset_id))
        for dataset_id in manifest["benchmark"]["regression_dataset_ids"]
    } | {
        ("canary_no_variance", int(dataset_id))
        for dataset_id in manifest["benchmark"]["categorical_canary_ids"]
    }
    found = set()
    for dataset in datasets:
        coordinate = (dataset["kind"], int(dataset["dataset_id"]))
        if coordinate in found:
            raise RuntimeError(f"duplicate raw dataset {coordinate}")
        found.add(coordinate)
        if not dataset.get("dataset_sha256"):
            raise RuntimeError(f"dataset {coordinate} lacks a content hash")
        splits = dataset.get("splits")
        if not splits or any(not row.get("indices_sha256") for row in splits):
            raise RuntimeError(f"dataset {coordinate} lacks split hashes")
        expected_seeds = (
            set(manifest["benchmark"]["split_seeds"])
            if coordinate[0] == "regression_ledger"
            else set(manifest["benchmark"]["canary_seeds"])
        )
        if (
            len(splits) != len(expected_seeds)
            or {row.get("seed") for row in splits} != expected_seeds
        ):
            raise RuntimeError(f"dataset {coordinate} has the wrong splits")
    if found != expected:
        raise RuntimeError("raw dataset boundary differs from the manifest")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    stored_fingerprint = manifest.get("run_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("run_fingerprint", None)
    actual_fingerprint = hashlib.sha256(
        runner._canonical_json(unsigned)
    ).hexdigest()
    if stored_fingerprint != actual_fingerprint:
        raise RuntimeError("raw manifest fingerprint does not reproduce")
    source = manifest.get("source", {})
    if (
        source.get("clean") is not True
        or source.get("branch") != "main"
        or source.get("commit") != source.get("origin_main")
    ):
        raise RuntimeError("formal runner source attestation failed")
    expected_paths = {
        "protocol_sha256": runner.PROTOCOL_PATH,
        "corpus_sha256": runner.CORPUS_PATH,
        "suites_sha256": runner.SUITES_PATH,
        "goldens_sha256": runner.GOLDENS_PATH,
        "freeze_sha256": runner.FREEZE_PATH,
        "runner_sha256": runner.RUNNER_PATH,
        "analyzer_sha256": runner.ANALYZER_PATH,
    }
    for field, path in expected_paths.items():
        expected = hashlib.sha256(
            _git_blob(source["commit"], path)
        ).hexdigest()
        if manifest["inputs"].get(field) != expected:
            raise RuntimeError(f"raw manifest input hash changed: {field}")


def _validate_shard_projection(raw: dict[str, Any]) -> None:
    manifest = raw["manifest"]
    declared = {}
    for entry in raw.get("shards", ()):
        coordinate = (entry["kind"], int(entry["dataset_id"]))
        if coordinate in declared:
            raise RuntimeError(f"duplicate raw shard declaration {coordinate}")
        declared[coordinate] = entry["sha256"]
    rebuilt = {}
    for dataset in raw["datasets"]:
        coordinate = (dataset["kind"], int(dataset["dataset_id"]))
        records = [
            record
            for record in raw["records"]
            if (
                record["kind"] == coordinate[0]
                and int(record["dataset_id"]) == coordinate[1]
            )
        ]
        shard = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_fingerprint": manifest["run_fingerprint"],
            "kind": dataset["kind"],
            "dataset_id": dataset["dataset_id"],
            "dataset_key": dataset["dataset_key"],
            "dataset_sha256": dataset["dataset_sha256"],
            "metadata": dataset["metadata"],
            "splits": dataset["splits"],
            "records": records,
        }
        rebuilt[coordinate] = hashlib.sha256(
            runner._canonical_json(shard)
        ).hexdigest()
    if declared != rebuilt:
        raise RuntimeError("raw shard hashes do not reproduce")


def _freeze_floor_check(source_commit: str) -> dict[str, Any]:
    freeze_bytes = _git_blob(source_commit, FREEZE_PATH)
    freeze = json.loads(freeze_bytes)
    records = {
        int(record["id"]): record for record in freeze["scan"]["records"]
    }
    details = {}
    passes = True
    for dataset_id in sorted(runner.CANARIES):
        record = records[dataset_id]
        values = np.asarray(record["ceiling_values"], dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            ok = False
        elif record["ceiling_metric"] == "excess_brier":
            ok = (
                float(np.mean(values)) <= 0.005
                and float(np.max(values)) <= 0.01
            )
        elif record["ceiling_metric"] == "rmse_ratio":
            ok = float(np.mean(values)) <= 1.1
        else:
            ok = False
        details[str(dataset_id)] = {
            "metric": record.get("ceiling_metric"),
            "values": values.tolist(),
            "mean": float(np.mean(values)),
            "worst": float(np.max(values)),
            "passes": bool(ok),
        }
        passes = passes and bool(ok)
    return {
        "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "n_canaries": len(details),
        "details": details,
        "passes": passes,
    }


def _decision(
    number: int,
    name: str,
    summary: dict[str, Any],
    agrees: bool,
    rule: str,
) -> dict[str, Any]:
    return {
        "number": number,
        "name": name,
        "rule": rule,
        "agrees": bool(agrees),
        "summary": summary,
    }


def analyze(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("artifact_kind") != "synthgen_darkofit_ledger_raw":
        raise RuntimeError("analyzer accepts only the raw SynthGen artifact")
    if "analysis" in raw or "scorecard" in raw:
        raise RuntimeError("raw boundary contains derived analysis fields")
    manifest = raw.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("raw artifact lacks its frozen manifest")
    _validate_manifest(manifest)
    if manifest.get("protected_outcome_sources_accessed") is not False:
        raise RuntimeError("protected outcome source attestation failed")
    if manifest["source"].get("clean") is not True:
        raise RuntimeError("formal runner source was not clean")
    _validate_dataset_boundary(raw, manifest)

    regression_ids = [
        int(value)
        for value in manifest["benchmark"]["regression_dataset_ids"]
    ]
    slices = {
        name: [int(value) for value in members]
        for name, members in manifest["benchmark"]["slices"].items()
    }
    if any(len(members) < 8 for members in slices.values()):
        raise RuntimeError("a frozen decision slice has fewer than 8 datasets")
    records = raw.get("records")
    if not isinstance(records, list):
        raise RuntimeError("raw artifact lacks coordinate records")
    regression = _record_map(
        records,
        kind="regression_ledger",
        metric="rmse",
        expected_ids=regression_ids,
        expected_seeds=tuple(manifest["benchmark"]["split_seeds"]),
        expected_arms=tuple(manifest["benchmark"]["config_order"]),
    )

    ordinary = slices["ordinary_regression"]
    noisy = slices["noisy_nonlinear"]
    smooth = slices["smooth_linear"]
    categorical = slices["categorical_regression"]

    d1 = pair_summary(regression, ordinary, runner.STUDENT_T, runner.CONTROL)
    d2 = pair_summary(regression, ordinary, runner.MAE, runner.CONTROL)
    d3 = pair_summary(
        regression, noisy, runner.RANDOM_STRENGTH_05, runner.CONTROL
    )
    d4 = pair_summary(
        regression,
        noisy,
        runner.RANDOM_STRENGTH_10,
        runner.RANDOM_STRENGTH_05,
    )
    d5 = pair_summary(
        regression, smooth, runner.LINEAR_LEAVES, runner.CONTROL
    )
    d6 = pair_summary(
        regression, smooth, runner.LINEAR_RESIDUAL, runner.LINEAR_LEAVES
    )
    d7 = pair_summary(regression, categorical, runner.TS4, runner.CONTROL)
    d8 = pair_summary(regression, ordinary, runner.ORDERED, runner.CONTROL)
    d9 = pair_summary(regression, ordinary, runner.CORE, runner.CONTROL)

    decisions = [
        _decision(
            1,
            "Student-t location versus RMSE",
            d1,
            d1["aggregate_ratio"] >= 1.001 and d1["losses"] > d1["wins"],
            "ratio >= 1.001 and losses > wins",
        ),
        _decision(
            2,
            "MAE versus RMSE",
            d2,
            d2["aggregate_ratio"] >= 1.001 and d2["losses"] > d2["wins"],
            "ratio >= 1.001 and losses > wins",
        ),
        _decision(
            3,
            "random_strength=0.5 versus 0.0",
            d3,
            d3["aggregate_ratio"] <= 0.999 and d3["wins"] > d3["losses"],
            "ratio <= 0.999 and wins > losses",
        ),
        _decision(
            4,
            "random_strength=1.0 versus 0.5",
            d4,
            d4["aggregate_ratio"] >= 0.999 or d4["wins"] <= d4["losses"],
            "ratio >= 0.999 or no win majority",
        ),
        _decision(
            5,
            "local linear leaves versus constant leaves",
            d5,
            d5["aggregate_ratio"] <= 0.99
            and d5["wins"] >= math.ceil(2.0 * d5["n_datasets"] / 3.0),
            "ratio <= 0.99 and wins on at least two thirds",
        ),
        _decision(
            6,
            "global linear residual versus local linear leaves",
            d6,
            d6["aggregate_ratio"] >= 1.01 and d6["losses"] > d6["wins"],
            "ratio >= 1.01 and losses > wins",
        ),
        _decision(
            7,
            "ts_permutations=4 versus 1",
            d7,
            d7["improvement"] < 0.005
            or d7["maximum_dataset_regression"] > 0.005,
            "improvement < 0.5% or a dataset regresses > 0.5%",
        ),
        _decision(
            8,
            "forced ordered boosting versus scalar default",
            d8,
            d8["improvement"] < 0.005
            or d8["maximum_dataset_regression"] > 0.02,
            "improvement < 0.5% or a dataset regresses > 2%",
        ),
        _decision(
            9,
            "frozen speed-oriented core profile",
            d9,
            d9["improvement"] < 0.005
            or d9["maximum_dataset_regression"] > 0.005
            or d9["maximum_split_regression"] > 0.02,
            "fails any original broad promotion gate",
        ),
    ]

    canary_ids = [
        int(value) for value in manifest["benchmark"]["categorical_canary_ids"]
    ]
    canary_metrics = _record_map(
        records,
        kind="canary_no_variance",
        metric="brier",
        expected_ids=canary_ids,
        expected_seeds=tuple(manifest["benchmark"]["canary_seeds"]),
        expected_arms=(runner.CONTROL, runner.TS4),
    )
    if len(records) != len(regression) + len(canary_metrics):
        raise RuntimeError("raw boundary contains an unexpected record kind")
    dataset_hashes = {
        (dataset["kind"], int(dataset["dataset_id"])): dataset["dataset_sha256"]
        for dataset in raw["datasets"]
    }
    for record in records:
        coordinate = (record["kind"], int(record["dataset_id"]))
        if record.get("dataset_sha256") != dataset_hashes[coordinate]:
            raise RuntimeError(
                f"raw record dataset hash mismatch at {coordinate}"
            )
    _validate_shard_projection(raw)
    canary_deltas = {}
    for dataset_id in canary_ids:
        canary_deltas[str(dataset_id)] = float(
            np.mean(
                [
                    canary_metrics[(dataset_id, seed, runner.TS4)]
                    - canary_metrics[(dataset_id, seed, runner.CONTROL)]
                    for seed in runner.CANARY_SEEDS
                ]
            )
        )
    canary_mean_delta = float(np.mean(list(canary_deltas.values())))
    freeze_floor = _freeze_floor_check(manifest["source"]["commit"])
    canary = {
        "dataset_ids": canary_ids,
        "dataset_mean_brier_deltas": canary_deltas,
        "equal_dataset_mean_brier_delta": canary_mean_delta,
        "no_variance_passes": canary_mean_delta <= runner.EPSILON,
        "freeze_floor": freeze_floor,
    }

    agreements = sum(decision["agrees"] for decision in decisions)
    gates = {
        "complete_raw_boundary": True,
        "all_slices_have_at_least_8_datasets": True,
        "freeze_floor_passes": freeze_floor["passes"],
        "canary_no_variance_passes": canary["no_variance_passes"],
        "at_least_7_of_9_decisions_agree": agreements >= 7,
        "protected_outcome_sources_not_accessed": True,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "synthgen_darkofit_ledger_analysis",
        "raw_path": str(raw_path.resolve()),
        "raw_sha256": _sha256(raw_path),
        "analyzer_sha256": _sha256(ANALYZER_PATH),
        "source": manifest["source"],
        "run_fingerprint": manifest["run_fingerprint"],
        "coordinate_counts": {
            "regression": len(regression),
            "canary": len(canary_metrics),
            "total": len(regression) + len(canary_metrics),
        },
        "slices": slices,
        "decisions": decisions,
        "agreement_count": agreements,
        "canary": canary,
        "adoption_gates": gates,
        "adopted_as_mechanism_probe": all(gates.values()),
        "scope": (
            "Mechanism-probe adoption only; this artifact cannot promote a "
            "DarkoFit parameter, preset, policy, default, or release claim."
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    verdict = (
        "ADOPTED as a mechanism probe"
        if analysis["adopted_as_mechanism_probe"]
        else "NOT ADOPTED"
    )
    lines = [
        "# DarkoFit SynthGen df1 ledger result",
        "",
        f"**Verdict: {verdict}.**",
        "",
        analysis["scope"],
        "",
        f"Agreement: **{analysis['agreement_count']}/9**. "
        f"Raw coordinates: **{analysis['coordinate_counts']['total']}**.",
        "",
        "| # | Decision | Aggregate ratio | W-L-T | Agreement |",
        "|---:|---|---:|---:|:---:|",
    ]
    for decision in analysis["decisions"]:
        summary = decision["summary"]
        lines.append(
            f"| {decision['number']} | {decision['name']} | "
            f"{summary['aggregate_ratio']:.6f} | "
            f"{summary['wins']}-{summary['losses']}-{summary['ties']} | "
            f"{'yes' if decision['agrees'] else 'no'} |"
        )
    lines += [
        "",
        "## Canary and adoption gates",
        "",
        f"Categorical-canary equal-dataset mean Brier delta "
        f"(`TS=4 - control`): "
        f"`{analysis['canary']['equal_dataset_mean_brier_delta']:+.12g}`.",
        "",
    ]
    for name, passed in analysis["adoption_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += [
        "",
        "## Provenance",
        "",
        f"- Source commit: `{analysis['source']['commit']}`",
        f"- Run fingerprint: `{analysis['run_fingerprint']}`",
        f"- Raw SHA-256: `{analysis['raw_sha256']}`",
        f"- Analyzer SHA-256: `{analysis['analyzer_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    analysis = analyze(args.raw.resolve())
    _atomic_json(args.output_json.resolve(), analysis)
    _atomic_text(args.output_markdown.resolve(), render_markdown(analysis))
    print(
        f"{analysis['agreement_count']}/9 agreements; "
        f"adopted={analysis['adopted_as_mechanism_probe']}; "
        f"raw={analysis['raw_sha256']}"
    )


if __name__ == "__main__":
    main()
