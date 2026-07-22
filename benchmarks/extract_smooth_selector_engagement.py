#!/usr/bin/env python3
"""Extract the spent B-1 selector-engagement evidence into one bound record."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from paired_evidence_contract import write_create_only
except ImportError:  # pragma: no cover
    from benchmarks.paired_evidence_contract import write_create_only


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
CURRENT_RAW_PATH = REPO_ROOT / "benchmarks/v011_compute_ladder_v3_raw.json"
CURRENT_MANIFEST_PATH = REPO_ROOT / "benchmarks/v011_compute_ladder_v3_manifest.json"
CURRENT_PER_DATASET_PATH = (
    REPO_ROOT / "benchmarks/v011_compute_ladder_v3_per_dataset.csv"
)

M2_COMPLETION_SHA256 = (
    "1fbd09e4e71e537d58479b4343e3269a1cb7d1a8b56e6f8d23a59aa4b96c4b5c"
)
M2_MANIFEST_SHA256 = (
    "c96b020c82a873091d984dc992eef6a78a8bb6607271c0f7c9921531fd97867c"
)
CURRENT_RAW_SHA256 = (
    "96f594da1a0ea885aa55d45636049d97b9b6e1a7f56d85679dfe879420636f79"
)
CURRENT_MANIFEST_SHA256 = (
    "01fbb053d1390c43758adc4f47da38e39b6beb53be26ed13548a5eb399d485d4"
)
CURRENT_PER_DATASET_SHA256 = (
    "546592592a3a70720fa214245451374982f2e17f783341c07fbe03b97682dd10"
)
M2_FRAMEWORK = "ChimeraBoost_c1_same_machine_primary_M_BAG_L1"
DATASETS = {
    "airfoil_self_noise": "363612",
    "physiochemical_protein": "363693",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise RuntimeError(f"selector-engagement input hash drifted: {path}")


def _load_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_hash(path, expected_sha256)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"selector-engagement JSON is not an object: {path}")
    return payload


def _load_verified_pickle(
    cache: Path, relative: str, attestation: Mapping[str, Any]
) -> dict[str, Any]:
    record = attestation.get("result_artifacts", {}).get(relative)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"M2 result is absent from its attestation: {relative}")
    path = cache / relative
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or file_sha256(path) != record.get("sha256")
    ):
        raise RuntimeError(f"M2 result failed its attested hash/size: {relative}")
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - exact bytes are hash-attested
    if not isinstance(payload, dict):
        raise RuntimeError(f"M2 result payload is not an object: {relative}")
    return payload


def _child_fits(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    try:
        children = payload["method_metadata"]["info"]["children_info"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("M2 ChimeraBoost child metadata is missing") from exc
    if not isinstance(children, Mapping) or len(children) != 8:
        raise RuntimeError("M2 ChimeraBoost result must contain eight child fits")
    fits = []
    for name in sorted(children):
        child = children[name]
        fit = child.get("comparator_fit") if isinstance(child, Mapping) else None
        if not isinstance(fit, Mapping) or fit.get("engine") != "chimeraboost":
            raise RuntimeError("M2 ChimeraBoost child fit metadata drifted")
        fits.append(fit)
    return fits


def _count_states(values: Sequence[Any]) -> dict[str, int]:
    counts = Counter(
        "true" if value is True else "false" if value is False else "null"
        for value in values
    )
    return {name: counts[name] for name in ("true", "false", "null")}


def summarize_m2_dataset(
    cache: Path,
    attestation: Mapping[str, Any],
    *,
    dataset: str,
    task_id: str,
) -> dict[str, Any]:
    prefix = f"experiments/data/{M2_FRAMEWORK}/{task_id}/"
    relatives = sorted(
        relative
        for relative in attestation.get("result_artifacts", {})
        if relative.startswith(prefix) and relative.endswith("/results.pkl")
    )
    if len(relatives) != 3:
        raise RuntimeError(f"M2 {dataset} must have exactly three result artifacts")

    fits: list[Mapping[str, Any]] = []
    artifacts = []
    coordinates = []
    for relative in relatives:
        payload = _load_verified_pickle(cache, relative, attestation)
        metadata = payload.get("task_metadata", {})
        if str(metadata.get("tid")) != task_id or metadata.get("name") != dataset:
            raise RuntimeError(f"M2 task identity drifted: {relative}")
        child_fits = _child_fits(payload)
        fits.extend(child_fits)
        record = attestation["result_artifacts"][relative]
        artifacts.append({
            "path": relative,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        })
        coordinates.append({
            "fold": metadata.get("fold"),
            "repeat": metadata.get("repeat"),
            "child_fit_count": len(child_fits),
        })

    resolved = [fit.get("resolved_params", {}) for fit in fits]
    if any(not isinstance(params, Mapping) for params in resolved):
        raise RuntimeError("M2 resolved parameter metadata drifted")
    best_iterations = [fit.get("best_iteration") for fit in fits]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in best_iterations):
        raise RuntimeError("M2 best-iteration metadata drifted")

    return {
        "dataset": dataset,
        "task_id": task_id,
        "result_count": len(relatives),
        "child_fit_count": len(fits),
        "coordinates": coordinates,
        "selected_lane_counts": dict(sorted(Counter(
            str(fit.get("selected_lane")) for fit in fits
        ).items())),
        "linear_leaves_selected": _count_states([
            fit.get("linear_leaves_selected") for fit in fits
        ]),
        "linear_selection_performed": _count_states([
            fit.get("linear_selection_performed") for fit in fits
        ]),
        "resolved_linear_leaves": _count_states([
            params.get("linear_leaves") for params in resolved
        ]),
        "resolved_cross_features_nonnull_count": sum(
            params.get("cross_features") is not None for params in resolved
        ),
        "resolved_cat_combinations": _count_states([
            params.get("cat_combinations") for params in resolved
        ]),
        "best_iteration_min": min(best_iterations),
        "best_iteration_max": max(best_iterations),
        "artifacts": artifacts,
    }


def _quality_ratios(path: Path, dataset: str) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["dataset"] == dataset
            and row["comparison_kind"] == "reference"
            and row["numerator_arm"] == "darkofit_v011_default"
            and row["denominator_arm"] == "chimeraboost_v020_default"
        ]
    if len(rows) != 1:
        raise RuntimeError(f"current compute-ladder D0/M0 row drifted: {dataset}")
    row = rows[0]
    return {
        "test_rmse_ratio": float(row["test_rmse_ratio"]),
        "fit_seconds_ratio": float(row["fit_seconds_ratio"]),
        "prediction_seconds_per_call_ratio": float(
            row["prediction_seconds_per_call_ratio"]
        ),
    }


def summarize_current_dataset(
    raw: Mapping[str, Any], *, dataset: str
) -> dict[str, Any]:
    rows = [
        row
        for row in raw.get("rows", ())
        if row.get("dataset") == dataset
        and row.get("engine") == "chimeraboost"
        and row.get("arm") == "chimeraboost_v020_default"
    ]
    if len(rows) != 3:
        raise RuntimeError(f"current {dataset} must have three default rows")
    members = []
    coordinates = []
    for row in sorted(rows, key=lambda item: (item["repeat"], item["fold"])):
        model = row.get("model", {})
        if not isinstance(model, Mapping):
            raise RuntimeError(f"current {dataset} model metadata drifted")
        current_members = model.get("members")
        if (
            model.get("member_count") != 1
            or not isinstance(current_members, list)
            or len(current_members) != 1
        ):
            raise RuntimeError(f"current {dataset} default must have one member")
        member = current_members[0]
        if not isinstance(member, Mapping):
            raise RuntimeError(f"current {dataset} member metadata drifted")
        members.append(member)
        coordinates.append({"fold": row["fold"], "repeat": row["repeat"]})
    return {
        "dataset": dataset,
        "row_count": len(rows),
        "member_count": len(members),
        "coordinates": coordinates,
        "linear_leaves_selected": _count_states([
            member.get("linear_leaves_selected") for member in members
        ]),
        "cross_features_selected": _count_states([
            member.get("cross_features_selected") for member in members
        ]),
        "cross_pair_count": [member.get("cross_pair_count") for member in members],
        "quality_and_cost_vs_chimeraboost": _quality_ratios(
            CURRENT_PER_DATASET_PATH, dataset
        ),
    }


def build_record(m2_cache: Path) -> dict[str, Any]:
    cache = m2_cache.expanduser().resolve()
    completion_path = cache / "completion_attestation.json"
    manifest_path = cache / "run_manifest.json"
    attestation = _load_json(completion_path, M2_COMPLETION_SHA256)
    m2_manifest = _load_json(manifest_path, M2_MANIFEST_SHA256)
    raw = _load_json(CURRENT_RAW_PATH, CURRENT_RAW_SHA256)
    current_manifest = _load_json(CURRENT_MANIFEST_PATH, CURRENT_MANIFEST_SHA256)
    _require_hash(CURRENT_PER_DATASET_PATH, CURRENT_PER_DATASET_SHA256)
    if raw.get("contract_id") != "v011-release-compute-ladder-20260722-v3":
        raise RuntimeError("current compute-ladder contract identity drifted")

    m2 = {
        dataset: summarize_m2_dataset(
            cache, attestation, dataset=dataset, task_id=task_id
        )
        for dataset, task_id in DATASETS.items()
    }
    current = {
        dataset: summarize_current_dataset(raw, dataset=dataset)
        for dataset in DATASETS
    }
    m2_chimera = m2_manifest.get("source", {}).get("chimeraboost", {})
    if m2_chimera.get("git_head") != "f14be606b641f1bf0dc92bb14b3951f1fe631c6b":
        raise RuntimeError("M2 ChimeraBoost source pin drifted")
    if current_manifest.get("chimeraboost_source", {}).get("commit") != (
        "7d48e053e5bd3c7aded1126871aeb0f1f6b84c46"
    ):
        raise RuntimeError("current ChimeraBoost source pin drifted")

    return {
        "schema_version": 1,
        "kind": "spent_selector_engagement_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "spent_descriptive_causal_check_not_shipping_evidence",
        "fresh_or_lockbox_accessed": False,
        "inputs": {
            "m2_completion_attestation": {
                "sha256": M2_COMPLETION_SHA256,
                "result_count": attestation.get("result_count"),
            },
            "m2_run_manifest": {"sha256": M2_MANIFEST_SHA256},
            "m2_chimeraboost_source": {
                "commit": m2_chimera.get("git_head"),
                "tree": m2_chimera.get("git_tree"),
                "describe": m2_chimera.get("git_describe"),
            },
            "current_raw": {"sha256": CURRENT_RAW_SHA256},
            "current_manifest": {"sha256": CURRENT_MANIFEST_SHA256},
            "current_per_dataset": {"sha256": CURRENT_PER_DATASET_SHA256},
            "current_chimeraboost_source": current_manifest["chimeraboost_source"],
        },
        "m2_v018_outer_bagged": m2,
        "current_v020_direct_default": current,
        "decision": {
            "two_dataset_selector_hypothesis_confirmed": False,
            "protein_selector_signature_confirmed": True,
            "airfoil_selector_signature_confirmed": False,
            "airfoil_is_current_default_deficit": False,
            "selector_campaign_scope": "protein_and_generic_smooth_process_class",
            "airfoil_disposition": "remove_from_selector_causal_claim_and_leave_configuration_representation_history_spent",
            "next_action": "fund_new_identity_automatic_selector_development_on_spent_data_only",
        },
        "limitations": [
            "This is metadata inspection of already-spent M2 and release-ladder artifacts; it creates no new quality evidence.",
            "The M2 outer bagged coordinates contain eight child fits each and are not 24 independent datasets.",
            "The current v0.20 member summary does not expose categorical-combination selection; the M2 resolved parameters report it false in all 48 inspected child fits.",
            "Default-on shipping still requires separately authorized, prospectively frozen Tier-D fresh evidence.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"selector-engagement output is create-only: {output}")
    record = build_record(args.m2_cache)
    write_create_only(
        output, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
