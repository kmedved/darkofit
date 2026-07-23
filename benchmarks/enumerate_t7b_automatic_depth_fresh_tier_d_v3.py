#!/usr/bin/env python3
"""Enumerate concrete, verified resources for P1-v3 before panel freeze."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import (
    build_t7b_automatic_depth_fresh_tier_d_registry as registry_builder,
)
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as v2_runner


ENUMERATION_ID = "t7b-automatic-depth-fresh-tier-d-v3-enumeration-v1-20260723"
V1_REGISTRY = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_contamination_registry.json"
)
PROTOCOL = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_v3_enumeration_protocol.md"
)
R2_PLAN = ROOT / "R2_PLAN.md"
CHIMERABOOST_ROOT = Path("/Users/konstantinmedvedovsky/code/chimeraboost")
DISCLOSURE_PATHS = {
    "R2_PLAN.md",
    "BEAT_CHIMERABOOST_PLAN.md",
    "COUNTERPUNCH_PLAN.md",
    "benchmarks/TESTING_LOG.md",
    "tests/test_t7b_automatic_depth_fresh_tier_d.py",
}
DISCLOSURE_PREFIXES = (
    "benchmarks/t7b_automatic_depth_fresh_tier_d_",
    "benchmarks/enumerate_t7b_automatic_depth_fresh_tier_d_v3.py",
)
REQUIRED_MODULES = ("numpy", "pandas", "sklearn", "numba", "openml")


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


def _source_state(repository: Path) -> dict[str, Any]:
    status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    return {
        "path": str(repository.resolve()),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def _is_disclosure_path(path: str) -> bool:
    return path in DISCLOSURE_PATHS or path.startswith(DISCLOSURE_PREFIXES)


def _history_hits(
    lineage: Mapping[str, Any],
    *,
    darkofit_head: str,
    chimeraboost_head: str,
) -> list[dict[str, Any]]:
    hits = []
    for repository_name, repository, revision in (
        ("darkofit", ROOT, darkofit_head),
        ("chimeraboost", CHIMERABOOST_ROOT, chimeraboost_head),
    ):
        for literal in [
            lineage["dataset_name"],
            *(lineage.get("aliases") or []),
        ]:
            paths = registry_builder._git_grep_literal(
                repository, revision, str(literal)
            )
            if repository_name == "darkofit":
                paths = [path for path in paths if not _is_disclosure_path(path)]
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
            ("openml_task_id", int(lineage["task_id"])),
            ("task_id", int(lineage["task_id"])),
            ("openml_dataset_id", int(lineage["dataset_id"])),
            ("dataset_id", int(lineage["dataset_id"])),
        ):
            paths = registry_builder._git_grep_identifier(
                repository, revision, key, value
            )
            if repository_name == "darkofit":
                paths = [path for path in paths if not _is_disclosure_path(path)]
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
    return hits


def _module_environment() -> dict[str, Any]:
    modules = {}
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        modules[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "path": str(Path(module.__file__).resolve()),
        }
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "modules": modules,
    }


def enumerate_resources() -> dict[str, Any]:
    source = _source_state(ROOT)
    if not source["clean"]:
        raise RuntimeError("enumeration requires a clean DarkoFit checkout")
    published_refs = sorted(
        line.strip()
        for line in _git(
            ROOT, "branch", "-r", "--contains", source["head"]
        ).splitlines()
        if line.strip()
    )
    if not published_refs:
        raise RuntimeError("enumeration harness commit is not published")
    chimera = _source_state(CHIMERABOOST_ROOT)

    registry = json.loads(V1_REGISTRY.read_text(encoding="utf-8"))
    expected_registry_hash = v2_runner.json_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    )
    if registry["registry_sha256"] != expected_registry_hash:
        raise RuntimeError("v1 registry self-hash is invalid")

    known, thresholds = v2_runner._known_fingerprints()
    rows = []
    for lineage in sorted(
        registry["lineages"],
        key=lambda row: (str(row["stratum"]), str(row["lineage_id"])),
    ):
        history_hits = _history_hits(
            lineage,
            darkofit_head=source["head"],
            chimeraboost_head=chimera["head"],
        )
        if history_hits:
            rows.append(
                {
                    "lineage_id": lineage["lineage_id"],
                    "dataset_name": lineage["dataset_name"],
                    "task_id": int(lineage["task_id"]),
                    "dataset_id": int(lineage["dataset_id"]),
                    "stratum": lineage["stratum"],
                    "branch": lineage["branch"],
                    "split_kind": lineage["split_kind"],
                    "status": "ineligible",
                    "reason": "repository-history contamination reference",
                    "history_hits": history_hits,
                    "resource_loaded": False,
                }
            )
            continue
        try:
            verified = v2_runner._preflight_lineage(lineage, known, thresholds)
        except v2_runner.EligibilityError as exc:
            rows.append(
                {
                    "lineage_id": lineage["lineage_id"],
                    "dataset_name": lineage["dataset_name"],
                    "task_id": int(lineage["task_id"]),
                    "dataset_id": int(lineage["dataset_id"]),
                    "stratum": lineage["stratum"],
                    "branch": lineage["branch"],
                    "split_kind": lineage["split_kind"],
                    "status": "ineligible",
                    "reason": str(exc),
                    "history_hits": [],
                    "resource_loaded": True,
                }
            )
            continue
        rows.append(
            {
                **verified,
                "status": "eligible",
                "history_hits": [],
                "resource_loaded": True,
                "prior_v1_v2_value_free_contact_possible": True,
            }
        )

    eligible = [row for row in rows if row["status"] == "eligible"]
    ineligible = [row for row in rows if row["status"] != "eligible"]
    return {
        "schema_version": 1,
        "enumeration_id": ENUMERATION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "enumeration_complete",
        "source": {
            "darkofit": source,
            "darkofit_published_refs": published_refs,
            "chimeraboost": chimera,
        },
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL),
            "r2_plan": sha256_file(R2_PLAN),
            "v1_registry_file": sha256_file(V1_REGISTRY),
            "runner": sha256_file(Path(__file__)),
            "v2_preflight_helpers": sha256_file(Path(v2_runner.__file__)),
        },
        "environment": _module_environment(),
        "candidate_pool": {
            "source_registry_id": registry["registry_id"],
            "source_registry_sha256": registry["registry_sha256"],
            "declared_identities": len(rows),
            "selection": "all concrete v1 primary and reserve identities, each evaluated independently",
        },
        "eligible_identity_count": len(eligible),
        "ineligible_identity_count": len(ineligible),
        "eligible_stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in eligible).items())
        ),
        "eligible_branch_counts": dict(
            sorted(Counter(row["branch"] for row in eligible).items())
        ),
        "eligible_group_safe_count": sum(
            row["split_kind"] == "group_hash_3fold" for row in eligible
        ),
        "identities": rows,
        "attestations": {
            "concrete_identities_enumerated_before_panel_freeze": True,
            "all_eligible_resources_loaded_in_frozen_environment": True,
            "all_eligible_history_and_fingerprint_checks_passed": True,
            "all_eligible_splits_and_branches_verified": True,
            "target_statistics_computed": False,
            "target_values_persisted": False,
            "model_fit_started": False,
            "candidate_or_control_outcomes_inspected": False,
            "abstract_slot_substitution_performed": False,
            "confirmation_panel_frozen": False,
            "fresh_inspection_spent": False,
            "tabarena_used": False,
            "ctr23_executed": False,
            "lockbox_used": False,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = enumerate_resources()
    v2_runner._write_create_only(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "enumeration_id": artifact["enumeration_id"],
                "eligible_identity_count": artifact["eligible_identity_count"],
                "ineligible_identity_count": artifact["ineligible_identity_count"],
                "eligible_stratum_counts": artifact["eligible_stratum_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
