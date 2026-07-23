#!/usr/bin/env python3
"""Build the target-blind T7b automatic-depth fresh registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_registry_declarations.json")
)
PROTOCOL = (
    ROOT / "benchmarks" / ("t7b_automatic_depth_fresh_tier_d_execution_protocol.md")
)
OWNER_ADDENDUM = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_owner_addendum_20260723.md")
)
POWER_CONTRACT = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_power_design_contract.json")
)
POWER_RESULT = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_power_design_result_20260723.json")
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_contamination_registry.json")
)
EXPECTED_STRATA = {
    "low_density_numeric",
    "low_density_categorical_or_grouped",
    "high_density_numeric",
    "high_density_categorical_or_grouped",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _git(repository: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def _git_grep_literal(repository: Path, revision: str, literal: str) -> list[str]:
    completed = subprocess.run(
        ["git", "grep", "-il", "-F", literal, revision, "--", "."],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "git grep failed")
    return sorted(line for line in completed.stdout.splitlines() if line)


def _git_grep_identifier(
    repository: Path, revision: str, key: str, value: int
) -> list[str]:
    # JSON files in the history use both compact and pretty spacing.
    pattern = (
        rf'"{re.escape(key)}"[[:space:]]*:[[:space:]]*{int(value)}' r"([,}[:space:]]|$)"
    )
    completed = subprocess.run(
        ["git", "grep", "-il", "-E", pattern, revision, "--", "."],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "git grep failed")
    return sorted(line for line in completed.stdout.splitlines() if line)


def _validate_declarations(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported registry declarations")
    rows = value.get("lineages")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RuntimeError("registry requires 32 primaries and 8 reserves")
    task_ids = [int(row["task_id"]) for row in rows]
    dataset_ids = [int(row["dataset_id"]) for row in rows]
    lineage_ids = [str(row["lineage_id"]) for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("registry task IDs must be unique")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise RuntimeError("registry dataset IDs must be unique")
    if len(set(lineage_ids)) != len(lineage_ids):
        raise RuntimeError("registry lineage IDs must be unique")

    primary = [row for row in rows if int(row["priority"]) == 0]
    reserves = [row for row in rows if int(row["priority"]) == 1]
    if len(primary) != 32 or len(reserves) != 8:
        raise RuntimeError("registry priority composition changed")
    for subset, expected in ((primary, 8), (reserves, 2)):
        counts = Counter(str(row["stratum"]) for row in subset)
        if set(counts) != EXPECTED_STRATA or any(
            count != expected for count in counts.values()
        ):
            raise RuntimeError(f"registry stratum composition changed: {dict(counts)}")
    grouped = [row for row in primary if row["split_kind"] == "group_hash_3fold"]
    if len(grouped) < 4 or any(
        row["stratum"] != "low_density_categorical_or_grouped"
        or not row["group_column"]
        for row in grouped
    ):
        raise RuntimeError("four low-density group-safe lineages are required")
    for row in rows:
        if row["branch"] not in {"depth_4", "depth_8"}:
            raise RuntimeError("registry branch changed")
        expected_branch = (
            "depth_4" if str(row["stratum"]).startswith("low_density") else "depth_8"
        )
        if row["branch"] != expected_branch:
            raise RuntimeError("registry stratum/branch mismatch")
        expected_family = (
            "categorical_or_grouped"
            if str(row["stratum"]).endswith("categorical_or_grouped")
            else "numeric"
        )
        if row["feature_family"] != expected_family:
            raise RuntimeError("registry feature-family mismatch")
        if row["split_kind"] not in {
            "row_hash_5fold",
            "group_hash_3fold",
        }:
            raise RuntimeError("unsupported split kind")

    slots = Counter((row["slot"], int(row["priority"])) for row in rows)
    if any(count != 1 for count in slots.values()):
        raise RuntimeError("duplicate slot/priority declaration")
    reserve_slots = Counter(str(row["slot"]) for row in reserves)
    if any(count != 1 for count in reserve_slots.values()):
        raise RuntimeError("reserve slot ordering is ambiguous")


def build() -> dict[str, Any]:
    declarations = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    _validate_declarations(declarations)
    metadata = declarations["metadata_selection"]
    darko_head = str(metadata["darkofit_prefreeze_head"])
    chimera_head = str(metadata["chimeraboost_head"])
    chimera_root = Path("/Users/konstantinmedvedovsky/code/chimeraboost").resolve()
    if _git(ROOT, "rev-parse", darko_head) != darko_head:
        raise RuntimeError("DarkoFit pre-freeze head is unavailable")
    if _git(chimera_root, "rev-parse", chimera_head) != chimera_head:
        raise RuntimeError("ChimeraBoost contamination head is unavailable")

    repositories = (
        ("darkofit", ROOT, darko_head),
        ("chimeraboost", chimera_root, chimera_head),
    )
    records: list[dict[str, Any]] = []
    for row in declarations["lineages"]:
        hits: list[dict[str, Any]] = []
        literals = [row["dataset_name"], *(row.get("aliases") or [])]
        for repository_name, repository, revision in repositories:
            for literal in literals:
                paths = _git_grep_literal(repository, revision, str(literal))
                if paths:
                    hits.append(
                        {
                            "kind": "repository_literal",
                            "repository": repository_name,
                            "literal": str(literal),
                            "paths": paths,
                        }
                    )
            for key, value in (
                ("openml_task_id", int(row["task_id"])),
                ("task_id", int(row["task_id"])),
                ("openml_dataset_id", int(row["dataset_id"])),
                ("dataset_id", int(row["dataset_id"])),
            ):
                paths = _git_grep_identifier(repository, revision, key, value)
                if paths:
                    hits.append(
                        {
                            "kind": "repository_identifier",
                            "repository": repository_name,
                            "key": key,
                            "value": value,
                            "paths": paths,
                        }
                    )
        records.append(
            {
                **row,
                "normalized_name": normalize_name(row["dataset_name"]),
                "target_blind_contamination_status": (
                    "eligible" if not hits else "excluded"
                ),
                "target_blind_exposure_hits": hits,
                "feature_target_near_lineage_status": (
                    "required_after_registry_freeze_before_launch"
                ),
            }
        )

    excluded = [
        row["lineage_id"]
        for row in records
        if row["target_blind_contamination_status"] != "eligible"
    ]
    if excluded:
        raise RuntimeError(
            "target-blind registry failed closed: " + ", ".join(excluded)
        )

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "registry_id": declarations["registry_id"],
        "contract_id": ("t7b-automatic-depth-fresh-tier-d-execution-v1-20260723"),
        "selection_boundary": metadata,
        "source_hashes": {
            "declarations": sha256_file(DECLARATIONS),
            "protocol": sha256_file(PROTOCOL),
            "owner_addendum": sha256_file(OWNER_ADDENDUM),
            "power_contract": sha256_file(POWER_CONTRACT),
            "power_result": sha256_file(POWER_RESULT),
            "builder": sha256_file(Path(__file__)),
        },
        "source_revisions": {
            "darkofit_prefreeze_head": darko_head,
            "chimeraboost_head": chimera_head,
        },
        "composition": {
            "primary_lineages": 32,
            "reserve_lineages": 8,
            "coordinates_per_active_lineage": 3,
            "primary_stratum_counts": dict(
                sorted(
                    Counter(
                        row["stratum"] for row in records if int(row["priority"]) == 0
                    ).items()
                )
            ),
            "reserve_stratum_counts": dict(
                sorted(
                    Counter(
                        row["stratum"] for row in records if int(row["priority"]) == 1
                    ).items()
                )
            ),
            "primary_group_safe_lineages": sum(
                int(row["priority"]) == 0 and row["split_kind"] == "group_hash_3fold"
                for row in records
            ),
        },
        "coordinate_design": declarations["coordinate_design"],
        "lineages": records,
        "attestations": {
            "openml_metadata_only_before_freeze": True,
            "feature_values_downloaded_before_freeze": False,
            "target_values_downloaded_before_freeze": False,
            "darkofit_history_checked_at_exact_revision": True,
            "chimeraboost_history_checked_at_exact_revision": True,
            "all_prior_campaign_repository_references_in_scope": True,
            "spent_sports_cycle_in_scope": True,
            "fixed_reserve_order": True,
            "outcome_driven_substitution_forbidden": True,
            "near_lineage_fingerprint_preflight_required": True,
            "lockbox_data_used": False,
        },
    }
    artifact["registry_sha256"] = sha256_json(artifact)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = build()
    rendered = canonical_json_bytes(artifact)
    if args.verify_existing:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            raise RuntimeError("registry does not reproduce byte-for-byte")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        args.output.write_bytes(rendered)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "registry_sha256": artifact["registry_sha256"],
                "primary_lineages": artifact["composition"]["primary_lineages"],
                "reserve_lineages": artifact["composition"]["reserve_lineages"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
