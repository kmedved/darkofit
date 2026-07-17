#!/usr/bin/env python3
"""Build the target-blind native-ordinal C2 registry and power record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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


DECLARATIONS = ROOT / "benchmarks" / "native_ordinal_c2_declarations.json"
PROTOCOL = ROOT / "benchmarks" / "native_ordinal_c2_protocol.md"
CTR_SNAPSHOT = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
CTR_DECLARATIONS = ROOT / "benchmarks" / "ctr23_contamination_sources.json"
FRESH_REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
FRESH_REGISTRY_V2 = ROOT / "benchmarks" / "fresh_confirmation_registry_v2.json"
POWER_SOURCE = (
    ROOT
    / "benchmarks"
    / "tabarena_regression_ordinal_confirmation_paired_splits.csv"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "native_ordinal_c2_registry.json"

POWER_SEED = 20260717
POWER_SIMULATIONS = 200_000
POWER_LINEAGES = 5
POWER_EFFECT_RETENTION = 0.25
POWER_MIN_PROBABILITY = 0.80
CONFIRM_MAX_GEOMEAN = 0.995
CONFIRM_MIN_WINS = 3
CONFIRM_MAX_LINEAGE = 1.02
CONFIRM_MAX_SPLIT = 1.05
CONFIRM_BOOTSTRAP_LEVEL = 0.95


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def _git_clean(path: Path) -> bool:
    return not _git("status", "--porcelain", cwd=path)


def _git_grep(
    repository: Path,
    revision: str,
    *,
    fixed: str | None = None,
    pattern: str | None = None,
) -> list[str]:
    if (fixed is None) == (pattern is None):
        raise ValueError("provide exactly one git-grep expression")
    command = ["git", "grep", "-l"]
    if fixed is not None:
        command.extend(["-F", fixed])
    else:
        command.extend(["-E", pattern])
    command.extend([revision, "--", "."])
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "git grep failed")
    return sorted(line for line in completed.stdout.splitlines() if line)


def _structured_repository_hits(
    repository: Path,
    revision: str,
    declaration: dict[str, Any],
) -> list[dict[str, Any]]:
    task_id = int(declaration["task_id"])
    dataset_id = int(declaration["dataset_id"])
    name = str(declaration["dataset_name"])
    expressions = (
        (
            "task_id",
            rf'"(openml_)?task_id"[[:space:]]*:[[:space:]]*{task_id}([^0-9]|$)',
        ),
        (
            "dataset_id",
            rf'"(openml_)?dataset_id"[[:space:]]*:[[:space:]]*{dataset_id}([^0-9]|$)',
        ),
    )
    hits = [
        {"kind": kind, "paths": _git_grep(repository, revision, pattern=pattern)}
        for kind, pattern in expressions
    ]
    quoted_name_paths = sorted(
        set(
            _git_grep(repository, revision, fixed=f'"{name}"')
            + _git_grep(repository, revision, fixed=f"'{name}'")
        )
    )
    hits.append({"kind": "quoted_dataset_name", "paths": quoted_name_paths})
    return [record for record in hits if record["paths"]]


def _task_data(task_id: int):
    import openml

    task = openml.tasks.get_task(task_id, download_data=False)
    dataset = openml.datasets.get_dataset(
        int(task.dataset_id),
        download_data=True,
        download_qualities=False,
    )
    X, _unused_target, categorical, names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    return task, dataset, X, list(categorical), list(names)


def _canonical_domain(values: Any) -> list[Any]:
    result = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, (str, bool, int)):
            result.append(value)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("non-finite ordinal domain value")
            result.append(numeric)
        else:
            raise TypeError(
                f"unsupported ordinal domain scalar {type(value).__name__}"
            )
    return result


def _model_categorical_indices(
    X: Any, openml_categorical: list[bool]
) -> tuple[list[int], list[int]]:
    if len(openml_categorical) != int(X.shape[1]):
        raise RuntimeError("OpenML categorical indicator length changed")
    inferred = []
    for index, is_categorical in enumerate(openml_categorical):
        if is_categorical:
            continue
        try:
            np.asarray(X.iloc[:, index], dtype=np.float64)
        except (TypeError, ValueError):
            inferred.append(int(index))
    categorical = sorted(
        {
            *inferred,
            *(
                int(index)
                for index, is_categorical in enumerate(openml_categorical)
                if is_categorical
            ),
        }
    )
    return categorical, inferred


def _feature_record(
    declaration: dict[str, Any],
    task_record: dict[str, Any],
) -> dict[str, Any]:
    task, dataset, X, categorical, names = _task_data(
        int(declaration["task_id"])
    )
    if int(task.dataset_id) != int(declaration["dataset_id"]):
        raise RuntimeError("declared OpenML dataset ID drifted")
    if str(dataset.name) != str(declaration["dataset_name"]):
        raise RuntimeError("declared OpenML dataset name drifted")
    if str(task.target_name) != str(declaration["target_name"]):
        raise RuntimeError("declared OpenML target drifted")
    if X.shape != (
        int(task_record["fingerprint"]["n_rows"]),
        int(task_record["fingerprint"]["n_features"]),
    ):
        raise RuntimeError("feature matrix shape differs from task fingerprint")
    if names != list(X.columns):
        raise RuntimeError("OpenML feature names and dataframe columns differ")
    if len(set(str(name) for name in names)) != len(names):
        raise RuntimeError("OpenML feature names are not unique")
    categorical_indices, inferred_categorical_indices = (
        _model_categorical_indices(X, categorical)
    )
    categorical_set = set(categorical_indices)

    declaration_map = declaration["ordinal_features"]
    if not isinstance(declaration_map, dict):
        raise TypeError("ordinal_features declaration must be a mapping")
    if str(task.target_name) in declaration_map:
        raise RuntimeError("target cannot be declared as an ordinal feature")

    ordered = []
    name_to_index = {str(name): index for index, name in enumerate(names)}
    for feature, categories in declaration_map.items():
        if feature not in name_to_index:
            raise RuntimeError(
                f"declared ordinal feature {feature!r} is absent"
            )
        index = name_to_index[feature]
        if index not in categorical_set:
            raise RuntimeError(
                f"declared ordinal feature {feature!r} is numeric"
            )
        declared = _canonical_domain(categories)
        observed = _canonical_domain(
            X.iloc[:, index].dropna().unique().tolist()
        )
        if len(declared) < 2 or len(declared) != len(set(map(repr, declared))):
            raise RuntimeError(
                f"declared ordinal domain for {feature!r} is invalid"
            )
        if set(map(repr, declared)) != set(map(repr, observed)):
            raise RuntimeError(
                f"declared ordinal domain for {feature!r} is incomplete"
            )
        ordered.append(
            {
                "feature": feature,
                "index": int(index),
                "dtype": str(X.dtypes.iloc[index]),
                "categories": declared,
                "observed_domain_sha256": ctr.sha256_json(
                    sorted(map(repr, observed))
                ),
                "missing_count": int(X.iloc[:, index].isna().sum()),
            }
        )

    return {
        "feature_names": [str(name) for name in names],
        "feature_names_sha256": ctr.sha256_json(
            [str(name) for name in names]
        ),
        "openml_categorical_indices": [
            int(index)
            for index, is_categorical in enumerate(categorical)
            if is_categorical
        ],
        "inferred_nonnumeric_categorical_indices": (
            inferred_categorical_indices
        ),
        "categorical_indices": categorical_indices,
        "categorical_policy": (
            "openml_indicator_union_target_blind_float64_cast_failures"
        ),
        "ordinal_features": ordered,
        "ordinal_feature_count": len(ordered),
        "target_values_inspected": False,
    }


def _bootstrap_upper(task_log_ratios: np.ndarray) -> np.ndarray:
    task_log_ratios = np.asarray(task_log_ratios, dtype=np.float64)
    if task_log_ratios.ndim != 2 or task_log_ratios.shape[1] != POWER_LINEAGES:
        raise ValueError("power bootstrap requires five task log ratios")
    counts = np.asarray(
        [
            values
            for values in itertools.product(
                range(POWER_LINEAGES + 1), repeat=POWER_LINEAGES
            )
            if sum(values) == POWER_LINEAGES
        ],
        dtype=np.float64,
    )
    if counts.shape != (126, POWER_LINEAGES):
        raise RuntimeError("five-task bootstrap composition count changed")
    factorial = math.factorial
    probabilities = np.asarray(
        [
            factorial(POWER_LINEAGES)
            / math.prod(factorial(int(value)) for value in row)
            / (POWER_LINEAGES**POWER_LINEAGES)
            for row in counts
        ],
        dtype=np.float64,
    )
    if not math.isclose(
        float(np.sum(probabilities)), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError("five-task bootstrap probabilities are invalid")
    means = task_log_ratios @ (counts / POWER_LINEAGES).T
    order = np.argsort(means, axis=1)
    ordered_probabilities = np.take_along_axis(
        np.broadcast_to(probabilities, means.shape), order, axis=1
    )
    cumulative = np.cumsum(ordered_probabilities, axis=1)
    positions = np.argmax(
        cumulative >= CONFIRM_BOOTSTRAP_LEVEL, axis=1
    )
    ordered_means = np.take_along_axis(means, order, axis=1)
    return ordered_means[np.arange(len(ordered_means)), positions]


def _power_analysis() -> dict[str, Any]:
    source: dict[str, list[float]] = {}
    with POWER_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["contrast_code"] != "O/B":
                continue
            ratio = float(row["test_rmse_ratio"])
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise RuntimeError("spent ordinal power ratio is invalid")
            source.setdefault(row["dataset"], []).append(ratio)
    if set(source) != {"airfoil_self_noise", "diamonds"}:
        raise RuntimeError("spent ordinal power lineages changed")
    if {name: len(values) for name, values in source.items()} != {
        "airfoil_self_noise": 27,
        "diamonds": 6,
    }:
        raise RuntimeError("spent ordinal power coordinate count changed")

    names = sorted(source)
    logs = [np.log(np.asarray(source[name], dtype=np.float64)) for name in names]
    rng = np.random.default_rng(POWER_SEED)
    passing = 0
    block_size = 10_000
    for start in range(0, POWER_SIMULATIONS, block_size):
        rows = min(block_size, POWER_SIMULATIONS - start)
        lineage_choices = rng.integers(
            0, len(names), size=(rows, POWER_LINEAGES)
        )
        draws = np.empty(
            (rows, POWER_LINEAGES, 3), dtype=np.float64
        )
        for source_index, values in enumerate(logs):
            mask = lineage_choices == source_index
            draws[mask, :] = rng.choice(
                values, size=(int(np.sum(mask)), 3)
            )
        draws *= POWER_EFFECT_RETENTION
        lineage_logs = np.mean(draws, axis=2)
        lineage_ratios = np.exp(lineage_logs)
        geomean = np.exp(np.mean(lineage_logs, axis=1))
        wins = np.count_nonzero(lineage_ratios < 1.0, axis=1)
        maximum = np.max(lineage_ratios, axis=1)
        split_maximum = np.max(np.exp(draws), axis=(1, 2))
        bootstrap_upper = _bootstrap_upper(lineage_logs)
        passing += int(
            np.count_nonzero(
                (geomean <= CONFIRM_MAX_GEOMEAN)
                & (wins >= CONFIRM_MIN_WINS)
                & (maximum <= CONFIRM_MAX_LINEAGE)
                & (split_maximum <= CONFIRM_MAX_SPLIT)
                & (bootstrap_upper < 0.0)
            )
        )
    probability = passing / POWER_SIMULATIONS
    return {
        "source_path": str(POWER_SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256_file(POWER_SOURCE),
        "source_lineages": names,
        "source_coordinate_counts": {
            name: len(source[name]) for name in names
        },
        "source_ratio_sha256": ctr.sha256_json(source),
        "seed": POWER_SEED,
        "simulations": POWER_SIMULATIONS,
        "simulated_lineages": POWER_LINEAGES,
        "splits_per_lineage": 3,
        "effect_retention": POWER_EFFECT_RETENTION,
        "gates": {
            "equal_lineage_geomean_ratio_at_most": CONFIRM_MAX_GEOMEAN,
            "one_sided_task_bootstrap_upper_strictly_below": 1.0,
            "task_bootstrap_level": CONFIRM_BOOTSTRAP_LEVEL,
            "task_bootstrap_method": (
                "exact_multinomial_count_vectors_higher"
            ),
            "minimum_lineage_wins": CONFIRM_MIN_WINS,
            "maximum_lineage_ratio": CONFIRM_MAX_LINEAGE,
            "maximum_split_ratio": CONFIRM_MAX_SPLIT,
        },
        "passing_simulations": passing,
        "pass_probability": probability,
        "minimum_required_probability": POWER_MIN_PROBABILITY,
        "passes": probability >= POWER_MIN_PROBABILITY,
    }


def _known_exposure(
    declarations: dict[str, Any],
) -> tuple[set[str], set[int], list[dict[str, Any]], dict[str, Any]]:
    chimera = fresh._chimera_exposure_catalog()
    ctr_snapshot = _load(CTR_SNAPSHOT)
    ctr_tasks = (
        ctr_snapshot["ctr23_tasks"] + ctr_snapshot["spent_source_tasks"]
    )
    prior = _load(FRESH_REGISTRY)
    prior_v2 = _load(FRESH_REGISTRY_V2)
    prior_ids = {int(row["task_id"]) for row in prior["tasks"]}
    prior_v2_ids = {int(row["task_id"]) for row in prior_v2["tasks"]}
    if prior_ids != prior_v2_ids:
        raise RuntimeError(
            "fresh confirmation v2 task membership differs from its parent"
        )
    prior_tasks = [row["task_record"] for row in prior["tasks"]]
    development_records = [
        ctr._task_record(int(row["task_id"]), include_splits=True)
        for row in declarations["development_tasks"]
    ]
    names = set(chimera["normalized_names"])
    ids = set(map(int, chimera["openml_dataset_ids"]))
    for task in [*ctr_tasks, *prior_tasks, *development_records]:
        names.add(str(task["normalized_name"]))
        ids.add(int(task["openml_dataset_id"]))
    return names, ids, [*ctr_tasks, *prior_tasks, *development_records], chimera


def _base_record(
    declaration: dict[str, Any],
    task_record: dict[str, Any],
    feature_record: dict[str, Any],
    *,
    evidence_tier: str,
) -> dict[str, Any]:
    return {
        "task_id": int(declaration["task_id"]),
        "dataset_id": int(declaration["dataset_id"]),
        "dataset_name": str(declaration["dataset_name"]),
        "normalized_name": str(task_record["normalized_name"]),
        "target_name": str(declaration["target_name"]),
        "lineage_cluster": str(declaration["lineage_cluster"]),
        "role": str(declaration["role"]),
        "evidence_tier": evidence_tier,
        "ordinal_features": declaration["ordinal_features"],
        "semantic_sources": declaration["semantic_sources"],
        "feature_record": feature_record,
        "task_record": task_record,
        "target_values_inspected": False,
        "target_statistics_used": False,
    }


def build() -> dict[str, Any]:
    declarations = _load(DECLARATIONS)
    if declarations.get("schema_version") != 1:
        raise RuntimeError("unsupported native-ordinal C2 declaration")
    head = _git("rev-parse", "HEAD")
    if head == declarations["darkofit_prefreeze_head"]:
        raise RuntimeError("registry builder must be committed before execution")
    if not _git_clean(ROOT):
        raise RuntimeError("native-ordinal C2 registry requires clean DarkoFit")
    if (
        _git("branch", "--show-current") != "main"
        or head != _git("rev-parse", "origin/main")
    ):
        raise RuntimeError(
            "native-ordinal C2 registry requires pushed DarkoFit main"
        )
    if (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                declarations["darkofit_prefreeze_head"],
                head,
            ],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("native-ordinal C2 prefreeze head is not an ancestor")
    if _git("rev-parse", "HEAD", cwd=CHIMERA_ROOT) != (
        declarations["chimeraboost_head"]
    ):
        raise RuntimeError("ChimeraBoost head differs from C2 declaration")
    if not _git_clean(CHIMERA_ROOT):
        raise RuntimeError(
            "native-ordinal C2 registry requires clean ChimeraBoost"
        )

    development_declarations = declarations["development_tasks"]
    confirmation_declarations = declarations["confirmation_tasks"]
    all_declarations = [
        *development_declarations,
        *confirmation_declarations,
    ]
    task_ids = [int(row["task_id"]) for row in all_declarations]
    dataset_ids = [int(row["dataset_id"]) for row in all_declarations]
    clusters = [str(row["lineage_cluster"]) for row in all_declarations]
    split_dimensions = declarations.get("official_split_dimensions")
    if (
        len(development_declarations) != 8
        or len(confirmation_declarations) != POWER_LINEAGES
        or len(set(task_ids)) != len(task_ids)
        or len(set(dataset_ids)) != len(dataset_ids)
        or len(set(clusters)) != len(clusters)
        or not isinstance(split_dimensions, dict)
        or set(split_dimensions) != {str(task_id) for task_id in task_ids}
    ):
        raise RuntimeError("native-ordinal C2 panel composition changed")

    task_records: dict[int, dict[str, Any]] = {}
    feature_records: dict[int, dict[str, Any]] = {}
    for declaration in all_declarations:
        task_id = int(declaration["task_id"])
        task_record = ctr._task_record(task_id, include_splits=True)
        if int(task_record["openml_dataset_id"]) != int(
            declaration["dataset_id"]
        ):
            raise RuntimeError(f"task {task_id} dataset ID drifted")
        if task_record["dataset_name"] != declaration["dataset_name"]:
            raise RuntimeError(f"task {task_id} dataset name drifted")
        if task_record["target_name"] != declaration["target_name"]:
            raise RuntimeError(f"task {task_id} target drifted")
        if int(task_record["openml_task_type_id"]) != 2:
            raise RuntimeError(f"task {task_id} is not regression")
        expected_dimensions = split_dimensions[str(task_id)]
        if task_record["official_splits"]["dimensions"] != expected_dimensions:
            raise RuntimeError(f"task {task_id} split shape changed")
        task_records[task_id] = task_record
        feature_records[task_id] = _feature_record(
            declaration, task_record
        )

    known_names, known_ids, exposure_tasks, chimera = _known_exposure(
        declarations
    )
    thresholds = _load(CTR_DECLARATIONS)["near_match_thresholds"]
    development = [
        _base_record(
            declaration,
            task_records[int(declaration["task_id"])],
            feature_records[int(declaration["task_id"])],
            evidence_tier="spent_development",
        )
        for declaration in development_declarations
    ]

    confirmation = []
    for declaration in confirmation_declarations:
        task_id = int(declaration["task_id"])
        task_record = task_records[task_id]
        reasons = []
        normalized = str(task_record["normalized_name"])
        name_hit = fresh._name_hit(normalized, sorted(known_names))
        if name_hit is not None:
            reasons.append({"kind": "known_name", "match": name_hit})
        if int(task_record["openml_dataset_id"]) in known_ids:
            reasons.append(
                {
                    "kind": "known_openml_dataset_id",
                    "match": int(task_record["openml_dataset_id"]),
                }
            )
        for repository, revision, label in (
            (
                ROOT,
                declarations["darkofit_prefreeze_head"],
                "darkofit",
            ),
            (
                CHIMERA_ROOT,
                declarations["chimeraboost_head"],
                "chimeraboost",
            ),
        ):
            hits = _structured_repository_hits(
                repository, revision, declaration
            )
            if hits:
                reasons.append(
                    {
                        "kind": "repository_reference",
                        "repository": label,
                        "matches": hits,
                    }
                )
        near_matches = []
        for source in exposure_tasks:
            evidence = ctr.near_match_evidence(
                task_record["fingerprint"],
                source["fingerprint"],
                **thresholds,
            )
            if evidence["ambiguous"]:
                near_matches.append(
                    {
                        "source_task_id": int(source["openml_task_id"]),
                        "source_dataset_name": source["dataset_name"],
                        **evidence,
                    }
                )
        if task_record["fingerprint"]["canonicalization_ambiguous"]:
            reasons.append({"kind": "canonicalization_ambiguous"})
        if near_matches:
            reasons.append(
                {"kind": "near_lineage_alarm", "matches": near_matches}
            )
        record = _base_record(
            declaration,
            task_record,
            feature_records[task_id],
            evidence_tier="target_unseen_confirmation",
        )
        record["status"] = "eligible" if not reasons else "excluded"
        record["exclusion_reasons"] = reasons
        confirmation.append(record)

    excluded = [
        row["task_id"] for row in confirmation if row["status"] != "eligible"
    ]
    if excluded:
        raise RuntimeError(
            f"native-ordinal C2 confirmation candidates excluded: {excluded}"
        )

    folds = [int(value) for value in declarations["coordinate_folds"]]
    if folds != [0, 1, 2]:
        raise RuntimeError("native-ordinal C2 coordinate folds changed")
    coordinates = {
        tier: [
            {
                "task_id": int(row["task_id"]),
                "repeat": 0,
                "fold": fold,
                "sample": 0,
            }
            for row in rows
            for fold in folds
        ]
        for tier, rows in (
            ("development", development),
            ("confirmation", confirmation),
        )
    }
    power = _power_analysis()
    if not power["passes"]:
        raise RuntimeError("native-ordinal C2 confirmation design is underpowered")

    artifact = {
        "schema_version": 1,
        "registry_name": declarations["registry_name"],
        "builder_source_sha256": _sha256_file(Path(__file__)),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "declarations_sha256": _sha256_file(DECLARATIONS),
        "source_artifacts": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in (
                CTR_SNAPSHOT,
                CTR_DECLARATIONS,
                FRESH_REGISTRY,
                FRESH_REGISTRY_V2,
                POWER_SOURCE,
            )
        },
        "sources": {
            "darkofit_execution_head": head,
            "darkofit_prefreeze_head": declarations[
                "darkofit_prefreeze_head"
            ],
            "chimeraboost_head": declarations["chimeraboost_head"],
        },
        "exposure_catalog": chimera,
        "known_normalized_name_count": len(known_names),
        "known_openml_dataset_id_count": len(known_ids),
        "development_task_count": len(development),
        "development_engaged_task_count": sum(
            bool(row["ordinal_features"]) for row in development
        ),
        "confirmation_task_count": len(confirmation),
        "confirmation_lineage_count": len(
            {row["lineage_cluster"] for row in confirmation}
        ),
        "coordinates": coordinates,
        "coordinate_counts": {
            tier: len(rows) for tier, rows in coordinates.items()
        },
        "development_tasks": development,
        "confirmation_tasks": confirmation,
        "power_analysis": power,
        "selection_used_target_statistics": False,
        "development_outcomes_inspected": False,
        "confirmation_outcomes_inspected": False,
        "lockbox_touched": False,
        "confirmation_run_authorized": False,
    }
    artifact["registry_sha256"] = ctr.sha256_json(artifact)
    return artifact


def _atomic_create(path: Path, payload: bytes) -> None:
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    temporary = None
    try:
        import os
        import tempfile

        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if descriptor is not None:
            import os

            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = build()
    payload = ctr.canonical_json_bytes(artifact)
    _atomic_create(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().absolute()),
                "registry_sha256": artifact["registry_sha256"],
                "development_tasks": artifact["development_task_count"],
                "confirmation_tasks": artifact["confirmation_task_count"],
                "power": artifact["power_analysis"]["pass_probability"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
