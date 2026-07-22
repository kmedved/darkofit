#!/usr/bin/env python3
"""Record selector engagement on the exact M6 v3 grid without quality data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import bench_compare_revisions as comparison
    import m6_quality_rule_v3 as rule
    import paired_evidence_contract as evidence
    from benchmark_adapters import (
        FitConfig,
        RevisionSpec,
        build_dataset,
        make_sample_weight,
        split_case,
    )
except ImportError:  # pragma: no cover
    from benchmarks import bench_compare_revisions as comparison
    from benchmarks import m6_quality_rule_v3 as rule
    from benchmarks import paired_evidence_contract as evidence
    from benchmarks.benchmark_adapters import (
        FitConfig,
        RevisionSpec,
        build_dataset,
        make_sample_weight,
        split_case,
    )


IDENTITY = "automatic-linear-selector-v2-m6-engagement-20260722"
MECHANISM_ID = "automatic_linear_selector_v2"
INSPECTION_INDEX = 1
RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
PROTOCOL_PATH = RUNNER_PATH.with_name(
    "automatic_linear_selector_v2_m6_engagement_companion.md"
)
SELECTOR_CONTRACT_PATH = RUNNER_PATH.with_name(
    "automatic_linear_selector_v2_development_contract.md"
)
COMPARISON_PATH = RUNNER_PATH.with_name("bench_compare_revisions.py")
EVIDENCE_PATH = RUNNER_PATH.with_name("paired_evidence_contract.py")
RULE_PATH = RUNNER_PATH.with_name("m6_quality_rule_v3.py")
WORKER_PREFIX = "AUTOMATIC_SELECTOR_ENGAGEMENT="

_SELECTOR_FIELDS = {
    "version",
    "requested",
    "fit_random_state_seed",
    "eligible",
    "resolved_linear_leaves",
    "final_booster_linear_leaves",
    "final_linear_leaves_active",
    "reason",
    "minimum_relative_improvement",
    "split",
    "constant_validation_rmse",
    "linear_validation_rmse",
    "relative_validation_improvement",
    "selection_fits",
    "selection_total_seconds",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if root != repository or not (root / "darkofit").is_dir():
        raise RuntimeError(f"not a DarkoFit Git root: {repository}")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def expected_identities() -> tuple[tuple[str, str, int, str], ...]:
    return tuple(
        (dataset, size, seed, weight_mode)
        for size in rule.SIZES
        for dataset in rule.DATASETS
        for seed in rule.SEEDS
        for weight_mode in rule.WEIGHT_MODES
    )


def _selector_metadata_wrapper(**kwargs) -> dict[str, Any]:
    row = _ORIGINAL_EVIDENCE_ROW_METADATA(**kwargs)
    metadata = json.loads(row["model_metadata"])
    selector = getattr(kwargs["model"], "automatic_linear_selector_", None)
    metadata["automatic_linear_selector"] = selector
    row["model_metadata"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    )
    return row


_ORIGINAL_EVIDENCE_ROW_METADATA = comparison.evidence_row_metadata


def _worker(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text())
    comparison.evidence_row_metadata = _selector_metadata_wrapper
    try:
        row = comparison._fit_worker(payload)
    finally:
        comparison.evidence_row_metadata = _ORIGINAL_EVIDENCE_ROW_METADATA
    print(WORKER_PREFIX + json.dumps(row, sort_keys=True, allow_nan=False))


def _run_worker(
    payload_path: Path, *, environment: Mapping[str, str]
) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--worker", str(payload_path)],
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
    )
    if process.returncode:
        raise RuntimeError(
            "selector engagement worker failed:\n"
            + (process.stderr.strip() or process.stdout.strip())
        )
    matches = [
        line[len(WORKER_PREFIX):]
        for line in process.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("selector engagement worker output is invalid")
    row = json.loads(matches[0])
    if row.get("status") != "ok" or row.get("error"):
        raise RuntimeError(
            f"selector engagement worker returned an error: {row.get('error')}"
        )
    return row


def _selector_record(row: Mapping[str, Any], *, task: str) -> dict[str, Any]:
    metadata = json.loads(str(row["model_metadata"]))
    selector = metadata.get("automatic_linear_selector")
    if task == "regression":
        if (
            not isinstance(selector, dict)
            or set(selector) != _SELECTOR_FIELDS
            or selector.get("requested") != "auto"
            or not isinstance(selector.get("eligible"), bool)
            or not isinstance(selector.get("resolved_linear_leaves"), bool)
            or not isinstance(selector.get("reason"), str)
            or not selector["reason"]
        ):
            raise RuntimeError("regression selector engagement metadata is invalid")
        return selector
    if selector is not None:
        raise RuntimeError("classification unexpectedly gained selector state")
    return {
        "eligible": False,
        "resolved_linear_leaves": False,
        "reason": "classification_not_applicable",
    }


def _external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    raise ValueError("engagement output must be outside the harness checkout")


def run(args: argparse.Namespace) -> Path:
    output = _external_output(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    candidate = args.candidate.expanduser().resolve()
    before = {
        "harness": source_state(REPO_ROOT),
        "candidate": source_state(candidate),
    }
    if any(not state["clean"] for state in before.values()):
        raise RuntimeError("selector engagement requires clean source trees")

    variant = RevisionSpec(
        "candidate_default", str(candidate), use_defaults=True
    )
    config = FitConfig(max_bins=128, threads=rule.THREADS)
    records = []
    with tempfile.TemporaryDirectory(
        prefix="darkofit-selector-engagement-"
    ) as directory:
        temporary = Path(directory)
        environment = evidence.fixed_worker_environment(
            temporary / "numba-cache", threads=rule.THREADS
        )
        for dataset, size, seed, weight_mode in expected_identities():
            spec, X, y, cat_features = build_dataset(dataset, size, seed)
            weights = make_sample_weight(y, spec.task, weight_mode)
            split = split_case(X, y, spec.task, seed, weights)
            data_path = temporary / (
                f"{dataset}-{size}-{seed}-{weight_mode}.npz"
            )
            comparison._save_case(data_path, split)
            payload = {
                "variant": asdict(variant),
                "fit_config": asdict(config),
                "data_path": str(data_path),
                "task": spec.task,
                "cat_features": cat_features,
                "seed": seed,
                "repeat": 1,
                "evidence_contract": evidence.CONTRACT_VERSION,
            }
            payload_path = temporary / (
                f"payload-{dataset}-{size}-{seed}-{weight_mode}.json"
            )
            payload_path.write_text(json.dumps(payload, sort_keys=True))
            worker = _run_worker(payload_path, environment=environment)
            base = comparison._base_row(
                variant, spec, size, seed, weight_mode, split, config
            )
            base["expected_class_count"] = comparison._expected_class_count(
                spec, split
            )
            base.update(worker)
            complete = comparison._complete_row(
                base, fields=comparison.EVIDENCE_CSV_FIELDS
            )
            evidence._validate_row(
                complete,
                expected_sources={"candidate_default": candidate},
                threads=rule.THREADS,
            )
            records.append({
                "dataset": dataset,
                "size": size,
                "seed": seed,
                "weight_mode": weight_mode,
                "task": spec.task,
                "case_sha256": worker["case_sha256"],
                "dataset_sha256": worker["dataset_sha256"],
                "split_sha256": worker["split_sha256"],
                "weight_sha256": worker["weight_sha256"],
                "selector": _selector_record(worker, task=spec.task),
            })
            print(
                f"ok {dataset:25s} seed={seed} weights={weight_mode}",
                flush=True,
            )

    if tuple(
        (record["dataset"], record["size"], record["seed"], record["weight_mode"])
        for record in records
    ) != expected_identities():
        raise RuntimeError("selector engagement grid is incomplete")
    after = {
        "harness": source_state(REPO_ROOT),
        "candidate": source_state(candidate),
    }
    if after != before:
        raise RuntimeError("selector engagement source changed during execution")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": IDENTITY,
        "mechanism_id": MECHANISM_ID,
        "inspection_index": INSPECTION_INDEX,
        "companion_only": True,
        "quality_metrics_recorded": False,
        "ranking_or_acceptance_authorized": False,
        "shipping_or_default_claim_authorized": False,
        "sources": before,
        "bindings": {
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "selector_contract_sha256": file_sha256(SELECTOR_CONTRACT_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "m6_rule_sha256": file_sha256(RULE_PATH),
            "comparison_runner_sha256": file_sha256(COMPARISON_PATH),
            "paired_evidence_sha256": file_sha256(EVIDENCE_PATH),
        },
        "execution": {
            "cell_count": len(records),
            "repeat_count": 1,
            "threads": rule.THREADS,
            "public_defaults": True,
            "fresh_worker_per_cell": True,
        },
        "records": records,
    }
    evidence.write_create_only(
        output,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    print(f"wrote selector engagement to {output}")
    print(f"artifact sha256: {file_sha256(output)}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker"]:
        if len(arguments) != 2:
            raise SystemExit("--worker requires one payload path")
        _worker(Path(arguments[1]))
        return
    run(parse_args(arguments))


if __name__ == "__main__":
    main()
