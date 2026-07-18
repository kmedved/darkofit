#!/usr/bin/env python3
"""Build the target-blind T5 composite registry and power record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import build_ctr23_contamination_registry as ctr  # noqa: E402
from benchmarks import build_fresh_confirmation_registry as fresh  # noqa: E402


DECLARATIONS = ROOT / "benchmarks" / "t5_composite_registry_declarations.json"
PROTOCOL = ROOT / "benchmarks" / "t5_composite_registry_protocol.md"
CTR_SNAPSHOT = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
CTR_DECLARATIONS = ROOT / "benchmarks" / "ctr23_contamination_sources.json"
FRESH_REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
FRESH_REGISTRY_V2 = ROOT / "benchmarks" / "fresh_confirmation_registry_v2.json"
ORDINAL_REGISTRY = ROOT / "benchmarks" / "native_ordinal_c2_registry.json"
ACCURACY_TABLE = (
    ROOT / "benchmarks" / "tabarena_regression_accuracy_shootout_per_dataset.csv"
)
CROSS_ANALYSIS = ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t5_composite_registry.json"

POWER_SEED = 20260717
POWER_SIMULATIONS = 200_000
POWER_LINEAGES = 25
MIN_POWER = 0.80
QUALITY_BAR = 0.995
UNCERTAINTY_BAR = 1.002
LOO_BAR = 0.998
HARM_BAR = 1.005
ONE_SIDED_95_Z = 1.6448536269514722
EXPECTED_STRATA = {
    "smooth_numeric": 9,
    "mixed_categorical": 9,
    "applied_noisy": 7,
}


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(path: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=path,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_head(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD")


def _git_clean(path: Path) -> bool:
    return not _git(path, "status", "--porcelain", "--untracked-files=all")


def _is_ancestor(path: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=path,
        check=False,
    )
    return completed.returncode == 0


def _task_records(payload: Any):
    records = []

    def visit(value):
        if isinstance(value, dict):
            record = value.get("task_record")
            if (
                isinstance(record, dict)
                and "openml_task_id" in record
                and "fingerprint" in record
            ):
                records.append(record)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    unique = {}
    for record in records:
        unique[int(record["openml_task_id"])] = record
    return list(unique.values())


def _spent_source_tasks():
    ctr_snapshot = _load(CTR_SNAPSHOT)
    records = list(
        ctr_snapshot["ctr23_tasks"] + ctr_snapshot["spent_source_tasks"]
    )
    for path in (FRESH_REGISTRY, FRESH_REGISTRY_V2, ORDINAL_REGISTRY):
        records.extend(_task_records(_load(path)))
    unique = {}
    for record in records:
        unique[int(record["openml_task_id"])] = record
    return list(unique.values())


def _verify_ordinal_domain(task_id: int, declarations: dict[str, list[Any]]):
    if not declarations:
        return {}
    import openml

    task = openml.tasks.get_task(task_id, download_splits=False)
    dataset = task.get_dataset()
    X, _y, _categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    verified = {}
    for feature, expected in declarations.items():
        if feature not in X.columns:
            raise RuntimeError(
                f"task {task_id} ordinal feature {feature!r} is absent"
            )
        observed = [
            value.item() if isinstance(value, np.generic) else value
            for value in X[feature].dropna().unique().tolist()
        ]
        if len(observed) != len(expected) or set(observed) != set(expected):
            raise RuntimeError(
                f"task {task_id} ordinal domain drifted for {feature!r}"
            )
        verified[feature] = {
            "categories": list(expected),
            "observed_domain_sha256": ctr.sha256_json(
                sorted(str(value) for value in observed)
            ),
            "target_values_inspected": False,
        }
    return verified


def _power_sources():
    rows = []
    with ACCURACY_TABLE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["contrast"] == "A10/M":
                rows.append(row)
    if len(rows) != 13 or {row["dataset"] for row in rows} != {
        "airfoil_self_noise",
        "Another-Dataset-on-used-Fiat-500",
        "concrete_compressive_strength",
        "diamonds",
        "Food_Delivery_Time",
        "healthcare_insurance_expenses",
        "houses",
        "miami_housing",
        "physiochemical_protein",
        "QSAR-TID-11",
        "QSAR_fish_toxicity",
        "superconductivity",
        "wine_quality",
    }:
        raise RuntimeError("A10 power-source lineage set changed")
    guarded = {}
    for row in rows:
        validation = float(row["val_rmse_ratio"])
        test = float(row["test_rmse_ratio"])
        if not all(math.isfinite(value) and value > 0 for value in (validation, test)):
            raise RuntimeError("A10 power-source ratio is invalid")
        guarded[row["dataset"]] = test if validation <= QUALITY_BAR else 1.0
    omitted = "diamonds"
    if guarded[omitted] != min(guarded.values()):
        raise RuntimeError("declared conservative A10 omission is no longer best")
    a10 = [
        {
            "source": "guarded_a10_over_product_default",
            "lineage": name,
            "ratio": float(ratio),
        }
        for name, ratio in sorted(guarded.items())
        if name != omitted
    ]

    cross_payload = _load(CROSS_ANALYSIS)
    nominee = cross_payload["analysis"]["nominee"]
    if float(nominee["minimum_validation_improvement"]) != 0.05:
        raise RuntimeError("smooth-cross power source is not the 5% nominee")
    cross = [
        {
            "source": "five_percent_smooth_cross_nominee",
            "lineage": name,
            "ratio": float(ratio),
        }
        for name, ratio in sorted(nominee["dataset_ratios"].items())
    ]
    profiles = a10 + cross
    if len(a10) != 12 or len(cross) != 3 or len(profiles) != 15:
        raise RuntimeError("T5 plausible-effect pool changed")
    if any(
        not math.isfinite(row["ratio"]) or row["ratio"] <= 0.0
        for row in profiles
    ):
        raise RuntimeError("T5 plausible-effect ratio is invalid")
    return profiles, {
        "guard_margin": QUALITY_BAR,
        "a10_lineage_count_before_omission": 13,
        "a10_omitted_lineage": omitted,
        "a10_omission_reason": (
            "conservative concentration sensitivity; power must not depend "
            "on the historical Diamonds outlier"
        ),
        "a10_retained_lineage_count": len(a10),
        "smooth_cross_lineage_count": len(cross),
    }


def power_analysis():
    profiles, construction = _power_sources()
    pool = np.asarray([row["ratio"] for row in profiles], dtype=np.float64)
    log_pool = np.log(pool)
    rng = np.random.default_rng(POWER_SEED)
    passing = 0
    block = 10_000
    for start in range(0, POWER_SIMULATIONS, block):
        count = min(block, POWER_SIMULATIONS - start)
        draws = rng.choice(
            log_pool, size=(count, POWER_LINEAGES), replace=True
        )
        means = draws.mean(axis=1)
        point = np.exp(means)
        upper = np.exp(
            means
            + ONE_SIDED_95_Z
            * draws.std(axis=1, ddof=1)
            / math.sqrt(POWER_LINEAGES)
        )
        loo = np.exp(
            (draws.sum(axis=1) - draws.min(axis=1))
            / (POWER_LINEAGES - 1)
        )
        worst = np.exp(draws.max(axis=1))
        passing += int(
            np.count_nonzero(
                (point <= QUALITY_BAR)
                & (upper <= UNCERTAINTY_BAR)
                & (loo <= LOO_BAR)
                & (worst <= HARM_BAR)
            )
        )
    probability = passing / POWER_SIMULATIONS
    return {
        "seed": POWER_SEED,
        "simulations": POWER_SIMULATIONS,
        "simulated_lineages": POWER_LINEAGES,
        "effect_profiles": profiles,
        "effect_pool_sha256": ctr.sha256_json(profiles),
        "effect_pool_construction": construction,
        "gates": {
            "equal_dataset_geomean_ratio_at_most": QUALITY_BAR,
            "normal_approximation_one_sided_95_upper_at_most": UNCERTAINTY_BAR,
            "least_favorable_leave_one_out_ratio_at_most": LOO_BAR,
            "worst_dataset_ratio_at_most": HARM_BAR,
        },
        "upper_bound_design_approximation": (
            "normal bootstrap approximation on dataset log ratios; execution "
            "uses the frozen 100000-replicate hierarchical percentile bootstrap"
        ),
        "passing_simulations": passing,
        "pass_probability": probability,
        "minimum_required_probability": MIN_POWER,
        "passes": probability >= MIN_POWER,
        "conditional_on_spent_development_effects": True,
    }


def build():
    declarations = _load(DECLARATIONS)
    if declarations.get("schema_version") != 1:
        raise RuntimeError("unsupported T5 registry declaration")
    current_head = _git_head(ROOT)
    prefreeze = declarations["darkofit_prefreeze_head"]
    if current_head == prefreeze:
        raise RuntimeError("registry builder must be committed before execution")
    if not _is_ancestor(ROOT, prefreeze, current_head):
        raise RuntimeError("DarkoFit execution head does not descend from prefreeze")
    if not _git_clean(ROOT):
        raise RuntimeError("T5 registry requires a clean DarkoFit tree")
    if _git_head(CHIMERA_ROOT) != declarations["chimeraboost_head"]:
        raise RuntimeError("ChimeraBoost head differs from the declaration")
    if not _git_clean(CHIMERA_ROOT):
        raise RuntimeError("T5 registry requires clean ChimeraBoost")

    declared = declarations["candidates"]
    task_ids = [int(row["task_id"]) for row in declared]
    dataset_ids = [int(row["dataset_id"]) for row in declared]
    clusters = [str(row["lineage_cluster"]) for row in declared]
    strata = [str(row["stratum"]) for row in declared]
    if (
        len(declared) != 25
        or len(set(task_ids)) != 25
        or len(set(dataset_ids)) != 25
        or len(set(clusters)) != 25
    ):
        raise RuntimeError("T5 requires 25 unique tasks, datasets, and lineages")
    counts = {name: strata.count(name) for name in EXPECTED_STRATA}
    if counts != EXPECTED_STRATA:
        raise RuntimeError(f"T5 stratum composition changed: {counts}")

    exposure = fresh._chimera_exposure_catalog()
    source_tasks = _spent_source_tasks()
    known_names = sorted(
        set(exposure["normalized_names"])
        | {str(task["normalized_name"]) for task in source_tasks}
    )
    known_dataset_ids = set(exposure["openml_dataset_ids"]) | {
        int(task["openml_dataset_id"]) for task in source_tasks
    }
    thresholds = _load(CTR_DECLARATIONS)["near_match_thresholds"]

    records = []
    for declaration in declared:
        task_id = int(declaration["task_id"])
        task = ctr._task_record(task_id, include_splits=True)
        if task["normalized_name"] != declaration["expected_normalized_name"]:
            raise RuntimeError(f"task {task_id} name drifted")
        if int(task["openml_dataset_id"]) != int(declaration["dataset_id"]):
            raise RuntimeError(f"task {task_id} dataset ID drifted")
        if str(task["target_name"]) != str(declaration["expected_target_name"]):
            raise RuntimeError(f"task {task_id} target drifted")
        if int(task["openml_task_type_id"]) != 2:
            raise RuntimeError(f"task {task_id} is not supervised regression")
        if task["official_splits"]["dimensions"] != {
            "repeats": 1,
            "folds": 10,
            "samples": 1,
        }:
            raise RuntimeError(f"task {task_id} split dimensions changed")

        reasons = []
        name_hit = fresh._name_hit(task["normalized_name"], known_names)
        if name_hit is not None:
            reasons.append({"kind": "known_name", "match": name_hit})
        if int(task["openml_dataset_id"]) in known_dataset_ids:
            reasons.append(
                {
                    "kind": "known_openml_dataset_id",
                    "match": int(task["openml_dataset_id"]),
                }
            )
        for repository, revision, label in (
            (ROOT, prefreeze, "darkofit"),
            (CHIMERA_ROOT, declarations["chimeraboost_head"], "chimeraboost"),
        ):
            matches = fresh._git_grep(
                repository, revision, str(task["dataset_name"])
            )
            if matches:
                reasons.append(
                    {
                        "kind": "repository_reference",
                        "repository": label,
                        "literal": str(task["dataset_name"]),
                        "paths": matches,
                    }
                )
        near_matches = []
        for source in source_tasks:
            evidence = ctr.near_match_evidence(
                task["fingerprint"], source["fingerprint"], **thresholds
            )
            if evidence["ambiguous"]:
                near_matches.append(
                    {
                        "source_task_id": int(source["openml_task_id"]),
                        **evidence,
                    }
                )
        if task["fingerprint"]["canonicalization_ambiguous"]:
            reasons.append({"kind": "canonicalization_ambiguous"})
        if near_matches:
            reasons.append(
                {"kind": "near_lineage_alarm", "matches": near_matches}
            )
        ordinal_record = _verify_ordinal_domain(
            task_id, declaration["ordinal_features"]
        )
        records.append(
            {
                "task_id": task_id,
                "dataset_id": int(task["openml_dataset_id"]),
                "dataset_name": task["dataset_name"],
                "normalized_name": task["normalized_name"],
                "target_name": task["target_name"],
                "lineage_cluster": declaration["lineage_cluster"],
                "stratum": declaration["stratum"],
                "related_task_ids": declaration["related_task_ids"],
                "semantic_sources": declaration["semantic_sources"],
                "ordinal_features": declaration["ordinal_features"],
                "ordinal_domain_record": ordinal_record,
                "status": "eligible" if not reasons else "excluded",
                "exclusion_reasons": reasons,
                "target_statistics_used": False,
                "confirmation_outcomes_inspected": False,
                "task_record": task,
            }
        )

    pairwise_alarms = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            evidence = ctr.near_match_evidence(
                records[left]["task_record"]["fingerprint"],
                records[right]["task_record"]["fingerprint"],
                **thresholds,
            )
            if evidence["ambiguous"]:
                pairwise_alarms.append(
                    {
                        "left_task_id": records[left]["task_id"],
                        "right_task_id": records[right]["task_id"],
                        **evidence,
                    }
                )
    excluded = [row["task_id"] for row in records if row["status"] != "eligible"]
    if excluded or pairwise_alarms:
        raise RuntimeError(
            "T5 candidates failed closed: "
            f"excluded={excluded}, pairwise={pairwise_alarms}"
        )

    folds = [int(value) for value in declarations["coordinate_folds"]]
    if folds != [0, 1, 2]:
        raise RuntimeError("T5 coordinate folds changed")
    coordinates = [
        {"task_id": task_id, "repeat": 0, "fold": fold, "sample": 0}
        for task_id in task_ids
        for fold in folds
    ]
    power = power_analysis()
    if not power["passes"]:
        raise RuntimeError("T5 confirmation design lacks 80% simulated power")

    artifact = {
        "schema_version": 1,
        "registry_name": declarations["registry_name"],
        "created_from_clean_sources": True,
        "builder_source_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "declarations_sha256": _sha256(DECLARATIONS),
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                CTR_SNAPSHOT,
                CTR_DECLARATIONS,
                FRESH_REGISTRY,
                FRESH_REGISTRY_V2,
                ORDINAL_REGISTRY,
                ACCURACY_TABLE,
                CROSS_ANALYSIS,
            )
        },
        "sources": {
            "darkofit_execution_head": current_head,
            "darkofit_prefreeze_head": prefreeze,
            "chimeraboost_head": declarations["chimeraboost_head"],
        },
        "exposure_catalog": exposure,
        "spent_source_task_count": len(source_tasks),
        "known_normalized_name_count": len(known_names),
        "known_openml_dataset_id_count": len(known_dataset_ids),
        "task_count": len(records),
        "lineage_count": len(set(clusters)),
        "stratum_counts": counts,
        "coordinate_count": len(coordinates),
        "coordinates": coordinates,
        "tasks": records,
        "pairwise_candidate_near_match_alarms": pairwise_alarms,
        "power_analysis": power,
        "confirmation_outcomes_inspected": False,
        "target_statistics_used": False,
        "lockbox_data_used": False,
        "confirmation_run_authorized": True,
        "default_promotion_authorized": False,
    }
    artifact["registry_sha256"] = ctr.sha256_json(artifact)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    artifact = build()
    args.output.write_bytes(ctr.canonical_json_bytes(artifact))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "registry_sha256": artifact["registry_sha256"],
                "task_count": artifact["task_count"],
                "coordinate_count": artifact["coordinate_count"],
                "power": artifact["power_analysis"]["pass_probability"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
