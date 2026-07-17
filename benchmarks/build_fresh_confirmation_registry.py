#!/usr/bin/env python3
"""Build the target-blind fresh confirmation registry and power design."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
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


DECLARATIONS = ROOT / "benchmarks" / "fresh_confirmation_registry_declarations.json"
PROTOCOL = ROOT / "benchmarks" / "fresh_confirmation_registry_protocol.md"
CTR_SNAPSHOT = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
CTR_DECLARATIONS = ROOT / "benchmarks" / "ctr23_contamination_sources.json"
SMOOTH_RESULT = ROOT / "benchmarks" / "smooth_group_linear_selector.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
POWER_SIMULATIONS = 200_000
POWER_SEED = 20260717
MIN_POWER = 0.8
POWER_SMOOTH_TASKS = 14
POWER_MIN_WINS = 9
POWER_MAX_GEOMEAN_RATIO = 0.98
POWER_MAX_LINEAGE_RATIO = 1.02


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _git_clean(path: Path) -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=path, text=True
    ).strip()


def _git_grep(path: Path, revision: str, literal: str) -> list[str]:
    completed = subprocess.run(
        ["git", "grep", "-l", "-F", literal, revision, "--", "."],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "git grep failed")
    return sorted(line for line in completed.stdout.splitlines() if line)


def _repository_literals(task: dict[str, Any]) -> tuple[str, ...]:
    return (str(task["dataset_name"]),)


def _load_chimera_benchmark_module():
    bench = CHIMERA_ROOT / "benchmarks"
    for path in (CHIMERA_ROOT, bench):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = bench / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location(
        "_darkofit_fresh_registry_chimera_benchmarks", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ChimeraBoost benchmark catalog")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} is absent from {path}")


def _chimera_exposure_catalog() -> dict[str, Any]:
    module = _load_chimera_benchmark_module()
    tabarena = _literal_assignment(
        CHIMERA_ROOT / "tests" / "test_highcard.py", "TABARENA_51"
    )
    names = set(module.OPENML_SUITE)
    names.update(
        name
        for values in module.GRINSZTAJN_DATASETS.values()
        for name in values
    )
    names.update(
        name
        for values in module.PMLB_DATASETS.values()
        for name, _task in values
    )
    names.update(module.HC_DATASETS)
    names.update(tabarena)
    dataset_ids = {
        int(spec["data_id"]) for spec in module.OPENML_SUITE.values()
    }
    dataset_ids.update(
        int(spec["data_id"]) for spec in module.HC_DATASETS.values()
    )
    sources = [
        CHIMERA_ROOT / "benchmarks" / "run_benchmarks.py",
        CHIMERA_ROOT / "tests" / "test_highcard.py",
        CHIMERA_ROOT / "benchmarks" / "synthgen" / "corpus_marginals.json",
    ]
    return {
        "normalized_names": sorted({ctr.normalize_name(name) for name in names}),
        "openml_dataset_ids": sorted(dataset_ids),
        "source_files": {
            str(path.relative_to(CHIMERA_ROOT)): _sha256(path)
            for path in sources
        },
        "tabarena_name_count": len(tabarena),
        "resolved_name_count": len(names),
    }


def _name_hit(name: str, known_names: list[str]) -> str | None:
    normalized = ctr.normalize_name(name)
    compact = normalized.replace("_", "")
    for known in known_names:
        other = known.replace("_", "")
        if compact == other:
            return known
        shorter, longer = sorted((compact, other), key=len)
        if len(shorter) >= 6 and shorter in longer:
            return known
    return None


def _power_analysis(smooth_artifact: dict[str, Any]) -> dict[str, Any]:
    contrast = smooth_artifact["analysis"]["contrasts"][
        "selector_over_default"
    ]
    ratios = [
        float(ratio)
        for task in contrast["per_dataset"].values()
        for ratio in task["split_ratios"]
    ]
    if len(ratios) != 21 or any(
        not math.isfinite(value) or value <= 0.0 for value in ratios
    ):
        raise RuntimeError("spent smooth effect distribution is invalid")
    rng = np.random.default_rng(POWER_SEED)
    passed = 0
    block = 10_000
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    for start in range(0, POWER_SIMULATIONS, block):
        count = min(block, POWER_SIMULATIONS - start)
        draws = rng.choice(
            log_ratios, size=(count, POWER_SMOOTH_TASKS), replace=True
        )
        geomean = np.exp(draws.mean(axis=1))
        wins = np.count_nonzero(draws < 0.0, axis=1)
        maximum = np.exp(draws.max(axis=1))
        passed += int(
            np.count_nonzero(
                (geomean <= POWER_MAX_GEOMEAN_RATIO)
                & (wins >= POWER_MIN_WINS)
                & (maximum <= POWER_MAX_LINEAGE_RATIO)
            )
        )
    probability = passed / POWER_SIMULATIONS
    return {
        "seed": POWER_SEED,
        "simulations": POWER_SIMULATIONS,
        "source_split_ratio_count": len(ratios),
        "source_split_ratios_sha256": ctr.sha256_json(ratios),
        "simulated_smooth_lineages": POWER_SMOOTH_TASKS,
        "gates": {
            "equal_lineage_geomean_ratio_at_most": POWER_MAX_GEOMEAN_RATIO,
            "minimum_lineage_wins": POWER_MIN_WINS,
            "maximum_lineage_ratio": POWER_MAX_LINEAGE_RATIO,
        },
        "passing_simulations": passed,
        "pass_probability": probability,
        "minimum_required_probability": MIN_POWER,
        "passes": probability >= MIN_POWER,
        "conditional_on_spent_three_lineage_development": True,
    }


def build() -> dict[str, Any]:
    declarations = _load(DECLARATIONS)
    if declarations.get("schema_version") != 1:
        raise RuntimeError("unsupported fresh registry declaration")
    if _git_head(ROOT) == declarations["darkofit_prefreeze_head"]:
        raise RuntimeError("registry builder must be committed before execution")
    if not _git_clean(ROOT):
        raise RuntimeError("fresh registry requires a clean DarkoFit tree")
    if _git_head(CHIMERA_ROOT) != declarations["chimeraboost_head"]:
        raise RuntimeError("ChimeraBoost head differs from the declaration")
    if not _git_clean(CHIMERA_ROOT):
        raise RuntimeError("fresh registry requires clean ChimeraBoost")

    exposure = _chimera_exposure_catalog()
    ctr_snapshot = _load(CTR_SNAPSHOT)
    thresholds = _load(CTR_DECLARATIONS)["near_match_thresholds"]
    source_tasks = (
        ctr_snapshot["ctr23_tasks"] + ctr_snapshot["spent_source_tasks"]
    )
    known_names = sorted(
        set(exposure["normalized_names"])
        | {str(task["normalized_name"]) for task in source_tasks}
    )
    known_dataset_ids = set(exposure["openml_dataset_ids"]) | {
        int(task["openml_dataset_id"]) for task in source_tasks
    }

    declared = declarations["candidates"]
    task_ids = [int(row["task_id"]) for row in declared]
    clusters = [str(row["lineage_cluster"]) for row in declared]
    if len(task_ids) != 20 or len(set(task_ids)) != 20:
        raise RuntimeError("fresh registry requires 20 unique primary tasks")
    if len(set(clusters)) != 20:
        raise RuntimeError("fresh registry primary lineages must be unique")
    strata = [str(row["stratum"]) for row in declared]
    if {
        name: strata.count(name)
        for name in ("smooth_numeric", "categorical", "noisy_tabular")
    } != {"smooth_numeric": 14, "categorical": 3, "noisy_tabular": 3}:
        raise RuntimeError("fresh registry stratum composition changed")

    records = []
    for declaration in declared:
        task_id = int(declaration["task_id"])
        task = ctr._task_record(task_id, include_splits=True)
        if task["normalized_name"] != declaration["expected_normalized_name"]:
            raise RuntimeError(f"task {task_id} name drifted")
        if int(task["openml_task_type_id"]) != 2:
            raise RuntimeError(f"task {task_id} is not supervised regression")
        dimensions = task["official_splits"]["dimensions"]
        if dimensions != {"repeats": 1, "folds": 10, "samples": 1}:
            raise RuntimeError(f"task {task_id} is not one-repeat 10-fold CV")

        reasons = []
        hit = _name_hit(task["normalized_name"], known_names)
        if hit is not None:
            reasons.append({"kind": "known_name", "match": hit})
        if int(task["openml_dataset_id"]) in known_dataset_ids:
            reasons.append(
                {
                    "kind": "known_openml_dataset_id",
                    "match": int(task["openml_dataset_id"]),
                }
            )
        for repository, revision, path in (
            (ROOT, declarations["darkofit_prefreeze_head"], "darkofit"),
            (CHIMERA_ROOT, declarations["chimeraboost_head"], "chimeraboost"),
        ):
            for literal in _repository_literals(task):
                matches = _git_grep(repository, revision, literal)
                if matches:
                    reasons.append(
                        {
                            "kind": "repository_reference",
                            "repository": path,
                            "literal": literal,
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
        records.append(
            {
                "task_id": task_id,
                "dataset_id": int(task["openml_dataset_id"]),
                "dataset_name": task["dataset_name"],
                "normalized_name": task["normalized_name"],
                "lineage_cluster": declaration["lineage_cluster"],
                "stratum": declaration["stratum"],
                "related_task_ids": declaration["related_task_ids"],
                "status": "eligible" if not reasons else "excluded",
                "exclusion_reasons": reasons,
                "task_record": task,
            }
        )

    excluded = [row["task_id"] for row in records if row["status"] != "eligible"]
    if excluded:
        raise RuntimeError(f"fresh registry candidates failed closed: {excluded}")

    coordinate_folds = [int(value) for value in declarations["coordinate_folds"]]
    coordinates = [
        {"task_id": task_id, "repeat": 0, "fold": fold, "sample": 0}
        for task_id in task_ids
        for fold in coordinate_folds
    ]
    power = _power_analysis(_load(SMOOTH_RESULT))
    if not power["passes"]:
        raise RuntimeError("fresh confirmation design lacks 80% simulated power")

    artifact = {
        "schema_version": 1,
        "registry_name": declarations["registry_name"],
        "created_from_clean_sources": True,
        "builder_source_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL),
        "declarations_sha256": _sha256(DECLARATIONS),
        "ctr_snapshot_sha256": _sha256(CTR_SNAPSHOT),
        "ctr_near_match_declarations_sha256": _sha256(CTR_DECLARATIONS),
        "smooth_development_artifact_sha256": _sha256(SMOOTH_RESULT),
        "sources": {
            "darkofit_execution_head": _git_head(ROOT),
            "darkofit_prefreeze_head": declarations["darkofit_prefreeze_head"],
            "chimeraboost_head": declarations["chimeraboost_head"],
        },
        "exposure_catalog": exposure,
        "known_normalized_name_count": len(known_names),
        "known_openml_dataset_id_count": len(known_dataset_ids),
        "task_count": len(records),
        "lineage_count": len(set(clusters)),
        "stratum_counts": {
            name: strata.count(name)
            for name in ("smooth_numeric", "categorical", "noisy_tabular")
        },
        "coordinate_count": len(coordinates),
        "coordinates": coordinates,
        "tasks": records,
        "power_analysis": power,
        "lockbox_data_used": False,
        "confirmation_data_scored": False,
        "selector_promotion_authorized": False,
        "lockbox_run_authorized": False,
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
