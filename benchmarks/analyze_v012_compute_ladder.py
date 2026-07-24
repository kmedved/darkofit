#!/usr/bin/env python3
"""Verify and analyze the DarkoFit v0.12 release compute ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import run_v012_compute_ladder as campaign


def _load_isolated_legacy_analysis() -> ModuleType:
    name = "benchmarks._v012_isolated_legacy_analysis"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = campaign.BENCH / "analyze_v011_compute_ladder.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy analysis helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


legacy_analysis = _load_isolated_legacy_analysis()
legacy_analysis.campaign = campaign
legacy_analysis.COUNTERPARTS = (
    (campaign.DARKO_DEFAULT, campaign.CHIMERA_DEFAULT),
    (campaign.DARKO_ACCURACY, campaign.CHIMERA_ACCURACY),
    (campaign.DARKO_ENSEMBLE, campaign.CHIMERA_ENSEMBLE),
)


def _load_json(path: Path, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant in {field}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{field} must be an object")
    return dict(payload)


def _verify_artifact(
    root: Path,
    record: Any,
    *,
    expected_path: str,
    field: str,
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{field} artifact fields are not exact")
    if record.get("path") != expected_path:
        raise RuntimeError(f"{field} artifact path drifted")
    candidate = Path(expected_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"{field} artifact path escapes the run")
    path = root / candidate
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{field} artifact is not a regular file")
    payload = path.read_bytes()
    if (
        type(record.get("bytes")) is not int
        or record["bytes"] != len(payload)
        or record.get("sha256") != campaign._sha256_bytes(payload)
    ):
        raise RuntimeError(f"{field} artifact digest drifted")
    return path, payload


def _expected_manifest_fields() -> set[str]:
    return {
        "schema_version",
        "kind",
        "run_id",
        "created_at_utc",
        "protocol",
        "runner",
        "analyzer",
        "harness_head",
        "darkofit_source",
        "chimeraboost_source",
        "tabarena_source",
        "latest_chimeraboost_release",
        "hardware",
        "exclusive_machine",
        "worker_environment",
        "expected_worker_count",
        "ordered_grid_sha256",
        "ordered_grid",
    }


def _verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if set(manifest) != _expected_manifest_fields():
        raise RuntimeError("compute-ladder manifest fields are not exact")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "v012_compute_ladder_manifest"
        or manifest.get("run_id") != campaign.RUN_ID
        or manifest.get("expected_worker_count") != campaign.EXPECTED_WORKERS
        or manifest.get("ordered_grid_sha256") != campaign.ordered_grid_sha256()
        or manifest.get("worker_environment") != campaign.WORKER_ENVIRONMENT
    ):
        raise RuntimeError("compute-ladder manifest identity drifted")
    expected_grid = [
        {
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
        }
        for dataset, repeat, fold, arm in campaign.expected_ordered_grid()
    ]
    if manifest.get("ordered_grid") != expected_grid:
        raise RuntimeError("compute-ladder manifest grid drifted")
    for field, expected_path, current_path in (
        (
            "protocol",
            str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)),
            campaign.PROTOCOL_PATH,
        ),
        (
            "runner",
            str(
                (campaign.ROOT / "benchmarks/run_v012_compute_ladder.py").relative_to(
                    campaign.ROOT
                )
            ),
            campaign.ROOT / "benchmarks/run_v012_compute_ladder.py",
        ),
        (
            "analyzer",
            str(campaign.ANALYZER_PATH.relative_to(campaign.ROOT)),
            campaign.ANALYZER_PATH,
        ),
    ):
        record = manifest.get(field)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or record.get("path") != expected_path
            or record.get("sha256") != campaign.sha256(current_path)
        ):
            raise RuntimeError(f"manifest {field} binding drifted")
    sources = campaign.validate_product_sources(
        Path(str(manifest["darkofit_source"]["path"])),
        Path(str(manifest["chimeraboost_source"]["path"])),
        Path(str(manifest["tabarena_source"]["path"])),
    )
    if sources != {
        "darkofit": manifest["darkofit_source"],
        "chimeraboost": manifest["chimeraboost_source"],
        "tabarena": manifest["tabarena_source"],
    }:
        raise RuntimeError("recorded product or data source drifted")
    latest = manifest.get("latest_chimeraboost_release")
    if (
        not isinstance(latest, Mapping)
        or latest.get("tag_name") != campaign.CHIMERABOOST_TAG
        or latest.get("published_at")
        != campaign.CHIMERABOOST_RELEASE_PUBLISHED_AT
        or latest.get("html_url")
        != "https://github.com/bbstats/chimeraboost/releases/tag/v0.23.0"
    ):
        raise RuntimeError("latest ChimeraBoost release attestation drifted")
    hardware = manifest.get("hardware")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("logical_cpus") != campaign.THREADS
        or hardware.get("physical_cpus") != campaign.THREADS
        or int(hardware.get("memory_bytes", 0)) <= 0
    ):
        raise RuntimeError("compute-ladder hardware identity drifted")
    exclusive = manifest.get("exclusive_machine")
    if (
        not isinstance(exclusive, Mapping)
        or exclusive.get("conflicting_benchmark_processes") != []
        or not isinstance(exclusive.get("load_average"), list)
    ):
        raise RuntimeError("exclusive-machine audit is invalid")
    head = manifest.get("harness_head")
    if not isinstance(head, str) or len(head) != 40:
        raise RuntimeError("recorded harness commit is invalid")
    current = campaign.legacy._git(campaign.ROOT, "rev-parse", "HEAD")
    campaign.legacy._git(
        campaign.ROOT,
        "merge-base",
        "--is-ancestor",
        head,
        current,
    )
    campaign.legacy._git(
        campaign.ROOT,
        "merge-base",
        "--is-ancestor",
        head,
        "origin/main",
    )
    return sources


def _verify_worker(
    raw: Any,
    *,
    expected_index: int,
    parent_pid: int | None,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"worker {expected_index} must be an object")
    row = dict(raw)
    if row.get("kind") != "v012_compute_ladder_worker":
        raise RuntimeError(f"worker {expected_index} kind drifted")
    adapted = dict(row)
    adapted["kind"] = "v011_compute_ladder_worker"
    normalized, observed_parent = legacy_analysis._verify_worker(
        adapted,
        expected_index=expected_index,
        parent_pid=parent_pid,
        manifest=manifest,
    )
    normalized["kind"] = "v012_compute_ladder_worker"
    return normalized, observed_parent


def verify_run(
    input_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(os.path.abspath(input_dir.expanduser()))
    manifest_path = root / "manifest.json"
    raw_path = root / "raw.json"
    terminal_path = root / "terminal.json"
    manifest = _load_json(manifest_path, "manifest")
    sources = _verify_manifest(manifest)
    terminal = _load_json(terminal_path, "terminal")
    if set(terminal) != {
        "schema_version",
        "kind",
        "status",
        "run_id",
        "completed_worker_count",
        "raw",
        "completed_at_utc",
    }:
        raise RuntimeError("compute-ladder terminal fields are not exact")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != "v012_compute_ladder_terminal"
        or terminal.get("status") != "complete"
        or terminal.get("run_id") != campaign.RUN_ID
        or terminal.get("completed_worker_count") != campaign.EXPECTED_WORKERS
    ):
        raise RuntimeError("compute-ladder terminal is not complete")
    _, raw_payload = _verify_artifact(
        root,
        terminal["raw"],
        expected_path="raw.json",
        field="raw",
    )
    raw = json.loads(raw_payload)
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "kind",
        "run_id",
        "started_at_utc",
        "completed_at_utc",
        "manifest",
        "workers",
        "rows",
    }:
        raise RuntimeError("compute-ladder raw fields are not exact")
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "v012_compute_ladder_raw"
        or raw.get("run_id") != campaign.RUN_ID
        or raw.get("completed_at_utc") != terminal.get("completed_at_utc")
    ):
        raise RuntimeError("compute-ladder raw identity drifted")
    _verify_artifact(
        root,
        raw["manifest"],
        expected_path="manifest.json",
        field="manifest",
    )
    worker_records = raw.get("workers")
    embedded_rows = raw.get("rows")
    if (
        not isinstance(worker_records, list)
        or not isinstance(embedded_rows, list)
        or len(worker_records) != campaign.EXPECTED_WORKERS
        or len(embedded_rows) != campaign.EXPECTED_WORKERS
    ):
        raise RuntimeError("compute-ladder worker count drifted")
    rows: list[dict[str, Any]] = []
    parent_pid: int | None = None
    for index, (record, embedded) in enumerate(zip(worker_records, embedded_rows)):
        _, payload = _verify_artifact(
            root,
            record,
            expected_path=f"workers/{index:03d}.json",
            field=f"worker {index}",
        )
        worker = json.loads(payload)
        if worker != embedded:
            raise RuntimeError(f"worker {index} embedded row drifted")
        normalized, parent_pid = _verify_worker(
            worker,
            expected_index=index,
            parent_pid=parent_pid,
            manifest=manifest,
        )
        rows.append(normalized)
    fingerprints: dict[tuple[str, int, int], Any] = {}
    dimensions: dict[tuple[str, int, int], tuple[int, int, int]] = {}
    for row in rows:
        key = (row["dataset"], row["repeat"], row["fold"])
        fingerprint = row["fingerprints"]
        shape = (row["train_rows"], row["test_rows"], row["feature_count"])
        if key in fingerprints and fingerprints[key] != fingerprint:
            raise RuntimeError(f"split fingerprints disagree within {key}")
        if key in dimensions and dimensions[key] != shape:
            raise RuntimeError(f"split dimensions disagree within {key}")
        fingerprints[key] = fingerprint
        dimensions[key] = shape
    if len(fingerprints) != campaign.EXPECTED_COORDINATES:
        raise RuntimeError("compute-ladder coordinate coverage drifted")
    provenance = {
        "run_id": campaign.RUN_ID,
        "harness_head": manifest["harness_head"],
        "protocol_sha256": manifest["protocol"]["sha256"],
        "runner_sha256": manifest["runner"]["sha256"],
        "analyzer_sha256": manifest["analyzer"]["sha256"],
        "manifest_sha256": campaign.sha256(manifest_path),
        "raw_sha256": campaign.sha256(raw_path),
        "terminal_sha256": campaign.sha256(terminal_path),
        "darkofit": sources["darkofit"],
        "chimeraboost": sources["chimeraboost"],
        "tabarena": sources["tabarena"],
    }
    return manifest, rows, provenance


def summarize(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paired = legacy_analysis.pair_rows(rows)
    summary, per_dataset = legacy_analysis.summarize(paired, provenance)
    summary["campaign"] = campaign.RUN_ID
    summary["decision"] = "descriptive_release_scoreboard"
    summary["strict_program_verdict"]["basis"] = "equal_dataset_point_estimates"
    summary["scope"]["latest_chimeraboost_release_at_worker_zero"] = (
        campaign.CHIMERABOOST_TAG
    )
    return summary, paired, per_dataset


def render_report(summary: Mapping[str, Any]) -> str:
    report = legacy_analysis.render_report(summary)
    replacements = {
        "# DarkoFit v0.11 release compute-ladder scoreboard": (
            "# DarkoFit v0.12 release compute-ladder scoreboard"
        ),
        "Status: **spent, descriptive release evidence; no policy advancement "
        "is authorized.**": (
            "Status: **descriptive release scoreboard; not a tuning or shipping "
            "gate.**"
        ),
        "pinned ChimeraBoost v0.20 default": "pinned ChimeraBoost v0.23 default",
        "the predeclared point-estimate readout": (
            "the fixed-protocol point-estimate readout"
        ),
    }
    for old, new in replacements.items():
        if old not in report:
            raise RuntimeError(f"legacy report template changed: {old}")
        report = report.replace(old, new)
    return report


def _write_create_only(path: Path, payload: bytes) -> None:
    legacy_analysis._write_create_only(path, payload)


def analyze(input_dir: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(input_dir.expanduser()))
    baseline = verify_run(root)
    _, rows, provenance = baseline
    summary, paired, per_dataset = summarize(rows, provenance)
    outputs = {
        root / "coordinate_ratios.csv": legacy_analysis._csv_bytes(
            paired, "coordinate ratios"
        ),
        root / "per_dataset.csv": legacy_analysis._csv_bytes(
            per_dataset, "per-dataset rows"
        ),
        root / "summary.json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        root / "report.md": render_report(summary).encode("utf-8"),
    }
    attestation_path = root / "analysis_attestation.json"
    if (
        attestation_path.exists()
        or attestation_path.is_symlink()
        or any(path.exists() or path.is_symlink() for path in outputs)
    ):
        raise RuntimeError("compute-ladder analysis outputs are create-only")
    if verify_run(root) != baseline:
        raise RuntimeError("compute-ladder run changed during analysis")
    written: list[Path] = []
    try:
        for path, payload in outputs.items():
            _write_create_only(path, payload)
            written.append(path)
        if verify_run(root) != baseline:
            raise RuntimeError("compute-ladder run changed while publishing")
        attestation = {
            "schema_version": 1,
            "kind": "v012_compute_ladder_analysis_attestation",
            "run_id": campaign.RUN_ID,
            "decision": "descriptive_release_scoreboard",
            "strict_pareto_victory": summary["strict_program_verdict"][
                "strict_pareto_victory"
            ],
            "input": {
                "raw_sha256": provenance["raw_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "terminal_sha256": provenance["terminal_sha256"],
                "analyzer_sha256": provenance["analyzer_sha256"],
            },
            "outputs": {
                path.name: campaign._stable_artifact(path, root)
                for path in outputs
            },
        }
        _write_create_only(
            attestation_path,
            (
                json.dumps(attestation, allow_nan=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        written.append(attestation_path)
    except BaseException:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=campaign.DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = analyze(parse_args(argv).input_dir)
    print(render_report(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
