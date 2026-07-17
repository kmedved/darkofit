#!/usr/bin/env python3
"""Create the no-score dtype-label amendment to fresh registry v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
PROTOCOL = ROOT / "benchmarks" / "fresh_confirmation_registry_v2_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "fresh_confirmation_registry_v2.json"
EXPECTED_V1_FILE_SHA256 = (
    "37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3"
)
EXPECTED_V1_REGISTRY_SHA256 = (
    "2d1f232e998d9f815a97f80735cbfebe5587c8b36f3fe246b26fbf355c4b5f64"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def build() -> dict:
    if _sha256(V1) != EXPECTED_V1_FILE_SHA256:
        raise RuntimeError("fresh registry v1 file changed")
    parent = json.loads(V1.read_text())
    if parent["registry_sha256"] != EXPECTED_V1_REGISTRY_SHA256:
        raise RuntimeError("fresh registry v1 canonical identity changed")
    if parent["confirmation_data_scored"]:
        raise RuntimeError("cannot amend registry after confirmation scoring")

    tasks = []
    profile_counts = {
        "numeric_only_complete": 0,
        "categorical_complete": 0,
        "categorical_with_missing": 0,
    }
    for row in parent["tasks"]:
        fingerprint = row["task_record"]["fingerprint"]
        has_categorical = bool(fingerprint["has_categorical"])
        has_missing = bool(fingerprint["has_missing_features"])
        if has_categorical and has_missing:
            profile = "categorical_with_missing"
        elif has_categorical:
            profile = "categorical_complete"
        elif has_missing:
            profile = "numeric_with_missing"
        else:
            profile = "numeric_only_complete"
        corrected = (
            "smooth_process"
            if row["stratum"] == "smooth_numeric"
            else row["stratum"]
        )
        if corrected == "smooth_process":
            if profile not in profile_counts:
                raise RuntimeError(
                    "primary smooth/process task has unexpected dtype profile"
                )
            profile_counts[profile] += 1
        tasks.append(
            {
                "task_id": int(row["task_id"]),
                "dataset_id": int(row["dataset_id"]),
                "lineage_cluster": row["lineage_cluster"],
                "v1_stratum": row["stratum"],
                "stratum": corrected,
                "feature_profile": profile,
                "has_categorical": has_categorical,
                "has_missing_features": has_missing,
                "status": row["status"],
            }
        )
    if profile_counts != {
        "numeric_only_complete": 5,
        "categorical_complete": 7,
        "categorical_with_missing": 2,
    }:
        raise RuntimeError("fresh primary dtype profile changed")

    artifact = {
        "schema_version": 2,
        "registry_name": "darkofit_fresh_confirmation_v2",
        "amendment_kind": "pre_score_descriptive_stratum_correction",
        "parent": {
            "path": str(V1.relative_to(ROOT)),
            "file_sha256": EXPECTED_V1_FILE_SHA256,
            "registry_sha256": EXPECTED_V1_REGISTRY_SHA256,
        },
        "protocol_sha256": _sha256(PROTOCOL),
        "builder_source_sha256": _sha256(Path(__file__)),
        "task_count": parent["task_count"],
        "lineage_count": parent["lineage_count"],
        "coordinate_count": parent["coordinate_count"],
        "coordinates_sha256": hashlib.sha256(
            _canonical_bytes(parent["coordinates"])
        ).hexdigest(),
        "stratum_counts": {
            "smooth_process": 14,
            "categorical": 3,
            "noisy_tabular": 3,
        },
        "smooth_process_feature_profile_counts": profile_counts,
        "tasks": tasks,
        "power_analysis": parent["power_analysis"],
        "contamination_decisions_changed": False,
        "coordinates_changed": False,
        "power_design_changed": False,
        "confirmation_data_scored": False,
        "selector_promotion_authorized": False,
        "lockbox_run_authorized": False,
    }
    artifact["registry_v2_sha256"] = hashlib.sha256(
        _canonical_bytes(artifact)
    ).hexdigest()
    return artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    artifact = build()
    args.output.write_bytes(_canonical_bytes(artifact))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "registry_v2_sha256": artifact["registry_v2_sha256"],
                "profile_counts": (
                    artifact["smooth_process_feature_profile_counts"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
