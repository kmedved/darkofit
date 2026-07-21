#!/usr/bin/env python3
"""Recompute the dated M3b r3 arm-vs-single development readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


QUALITY_SHA256 = (
    "5fec218cbc0ec97ef4b3fec10f65a89131a377cf026dbb80da809d6396ead6c3"
)
RESULT_SHA256 = (
    "3e6d0750e772c156b6c4daed948eb6baa640564ce87fe1ffee7414b3fe03c8bc"
)
CAMPAIGN_NAME = "wave2_m3b_ensemble_v3_r3_20260720"
READOUT_NAME = "wave2_m3b_r3_vs_single_readout_20260721"
FROZEN_DISPOSITION = "close_b1_b2_preserve_existing_opt_in"
ARMS = (
    "control",
    "b1_sampling",
    "b2_member_policy",
    "b1_b2_combined",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"bound artifact must be a regular file: {source}")
    if _sha256_path(source) != expected_sha256:
        raise ValueError(f"bound artifact hash differs: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bound artifact is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"bound artifact must contain an object: {source}")
    return payload


def _geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("loss ratios must be positive finite numbers")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _validated_quality_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if (
        payload.get("name") != CAMPAIGN_NAME
        or payload.get("phase") != "quality"
        or payload.get("status") != "complete"
        or not isinstance(rows, list)
        or len(rows) != 65
    ):
        raise ValueError("M3b r3 quality artifact identity is invalid")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    case_contract: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("M3b r3 quality row must be an object")
        case_id = row.get("case_id")
        arm = row.get("arm")
        domain = row.get("domain")
        metric = row.get("primary_metric")
        loss = row.get("primary_loss")
        if (
            not isinstance(case_id, str)
            or arm not in {"single_reference", *ARMS}
            or domain not in {"general", "sports"}
            or not isinstance(metric, str)
            or row.get("phase") != "quality"
            or row.get("repeat") != 0
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) <= 0.0
        ):
            raise ValueError("M3b r3 quality row is invalid")
        key = (case_id, arm)
        if key in by_key:
            raise ValueError("M3b r3 quality rows are not unique")
        contract = (domain, metric)
        if case_id in case_contract and case_contract[case_id] != contract:
            raise ValueError("M3b r3 case semantics differ across arms")
        case_contract[case_id] = contract
        by_key[key] = row

    cases = sorted(case_contract)
    if (
        len(cases) != 13
        or sum(case_contract[case][0] == "sports" for case in cases) != 9
        or sum(case_contract[case][0] == "general" for case in cases) != 4
        or any(
            (case, arm) not in by_key
            for case in cases
            for arm in ("single_reference", *ARMS)
        )
        or any(
            case_contract[case] != ("sports", "cold_player_rmse")
            for case in cases
            if case_contract[case][0] == "sports"
        )
    ):
        raise ValueError("M3b r3 quality grid is incomplete or has wrong semantics")
    return {f"{case}\0{arm}": row for (case, arm), row in by_key.items()}


def build_readout(
    quality_path: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    quality = _load_bound_json(quality_path, QUALITY_SHA256)
    result = _load_bound_json(result_path, RESULT_SHA256)
    rows = _validated_quality_rows(quality)
    if (
        result.get("name") != CAMPAIGN_NAME
        or result.get("quality_artifact_sha256") != QUALITY_SHA256
        or result.get("disposition") != FROZEN_DISPOSITION
        or result.get("retained_private_arms") != []
    ):
        raise ValueError("M3b r3 frozen result identity is invalid")
    candidates = result.get("candidates")
    combined = (
        candidates.get("b1_b2_combined")
        if isinstance(candidates, Mapping)
        else None
    )
    if (
        not isinstance(combined, Mapping)
        or combined.get("survives") is not False
        or combined.get("checks", {}).get("archive_to_single") is not False
    ):
        raise ValueError("M3b r3 combined-arm frozen disposition is invalid")
    archive_ratio = combined.get("resources", {}).get(
        "median_archive_to_single"
    )
    if (
        isinstance(archive_ratio, bool)
        or not isinstance(archive_ratio, (int, float))
        or float(archive_ratio) <= 4.0
    ):
        raise ValueError("M3b r3 archive-gate evidence is invalid")

    cases = sorted({key.split("\0", 1)[0] for key in rows})
    summaries = {}
    for arm in ARMS:
        ratios = {}
        for case in cases:
            single = float(rows[f"{case}\0single_reference"]["primary_loss"])
            candidate = float(rows[f"{case}\0{arm}"]["primary_loss"])
            ratios[case] = candidate / single
        all_values = list(ratios.values())
        sports_values = [
            ratio
            for case, ratio in ratios.items()
            if rows[f"{case}\0{arm}"]["domain"] == "sports"
        ]
        general_values = [
            ratio
            for case, ratio in ratios.items()
            if rows[f"{case}\0{arm}"]["domain"] == "general"
        ]
        summaries[arm] = {
            "all_case_geometric_mean": _geometric_mean(all_values),
            "sports_geometric_mean": _geometric_mean(sports_values),
            "general_geometric_mean": _geometric_mean(general_values),
            "worst_case_ratio": max(all_values),
            "wins_vs_single": sum(value < 1.0 for value in all_values),
            "case_count": len(all_values),
            "per_case_primary_ratio": ratios,
        }

    combined_summary = summaries["b1_b2_combined"]
    return {
        "schema_version": 1,
        "name": READOUT_NAME,
        "dated": "2026-07-21",
        "evidence_scope": "spent_post_hoc_development_readout",
        "amends_frozen_m3b_result": False,
        "quality_artifact": {
            "path": f"benchmarks/{Path(quality_path).name}",
            "sha256": QUALITY_SHA256,
        },
        "frozen_result_artifact": {
            "path": f"benchmarks/{Path(result_path).name}",
            "sha256": RESULT_SHA256,
        },
        "sports_primary_scope": (
            "player-disjoint cold-player rows within the frozen held-team view"
        ),
        "arms_vs_single": summaries,
        "finding": {
            "combined_beats_single_all_cases": (
                combined_summary["wins_vs_single"]
                == combined_summary["case_count"]
            ),
            "combined_case_count": combined_summary["case_count"],
            "combined_median_archive_to_single": float(archive_ratio),
            "frozen_archive_to_single_limit": 4.0,
            "combined_survived_frozen_gate": False,
            "frozen_disposition": FROZEN_DISPOSITION,
            "serialization_authorized": False,
        },
    }


def render_markdown(readout: Mapping[str, Any]) -> str:
    summaries = readout["arms_vs_single"]
    labels = {
        "control": "control (existing group8)",
        "b1_sampling": "b1_sampling",
        "b2_member_policy": "b2_member_policy",
        "b1_b2_combined": "b1_b2_combined",
    }
    lines = [
        "# M3b r3 arm-vs-single development readout — 2026-07-21",
        "",
        "This is a dated, hash-bound post-hoc readout from the immutable r3 "
        "quality artifact. It does not amend the frozen M3b result or gates.",
        "",
        "| Arm | Pooled | Sports cold-player | General | Worst | Wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        lines.append(
            f"| {labels[arm]} | "
            f"{summary['all_case_geometric_mean']:.6f} | "
            f"{summary['sports_geometric_mean']:.6f} | "
            f"{summary['general_geometric_mean']:.6f} | "
            f"{summary['worst_case_ratio']:.6f} | "
            f"{summary['wins_vs_single']}/{summary['case_count']} |"
        )
    finding = readout["finding"]
    lines.extend([
        "",
        "The combined arm beat the matched single on all 13 development cases. "
        "Its quality payload is therefore promising development evidence, but it "
        "did not survive the prospectively frozen campaign: median archive size "
        f"was {finding['combined_median_archive_to_single']:.6f}x single against "
        f"the unchanged <= {finding['frozen_archive_to_single_limit']:.1f}x gate.",
        "",
        "The nine sports primaries are player-disjoint cold-player rows within "
        "the frozen held-team view. The four general cases use the seeded 75/25 "
        "development split.",
        "",
        f"Frozen disposition: `{finding['frozen_disposition']}`. No serializer, "
        "public/default surface, fresh confirmation, or lockbox access is "
        "authorized by this readout.",
        "",
        "Bound artifacts:",
        "",
        f"- quality: `{readout['quality_artifact']['sha256']}`",
        f"- frozen result: `{readout['frozen_result_artifact']['sha256']}`",
        "",
    ])
    return "\n".join(lines)


def _parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        type=Path,
        default=root / "m3b_ensemble_v3_r3_quality.json",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=root / "m3b_ensemble_v3_r3_result.json",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    readout = build_readout(args.quality, args.result)
    rendered = (
        render_markdown(readout)
        if args.format == "markdown"
        else json.dumps(readout, indent=2, sort_keys=True) + "\n"
    )
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
