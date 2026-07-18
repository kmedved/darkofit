#!/usr/bin/env python3
"""Record the fail-closed T5 execution without fitting another model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_t5_composite_confirmation as runner  # noqa: E402


DEFAULT_SPOOL_DIRECTORY = runner.DEFAULT_SPOOL_DIRECTORY
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "t5_composite_confirmation_failure.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "benchmarks" / "t5_composite_confirmation_failure.md"
)
EXPECTED_DARKOFIT_HEAD = "da6881ecf1f58f251c9b3a6486c03000126d292c"
INVALID_TARGETS = {
    362367: {
        "dataset_id": 43462,
        "dataset_name": "Riga-real-estate-dataset",
        "target_name": "price",
        "rows": 4_689,
        "finite": 4_219,
        "nonfinite": 470,
        "nan": 470,
        "posinf": 0,
        "neginf": 0,
    },
    362395: {
        "dataset_id": 43853,
        "dataset_name": "Nintendo3DS-Games",
        "target_name": "metacritic",
        "rows": 1_680,
        "finite": 138,
        "nonfinite": 1_542,
        "nan": 1_542,
        "posinf": 0,
        "neginf": 0,
    },
}


def _git_head(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _target_finiteness(task_id):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    _X, y, _categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    return {
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "rows": int(len(values)),
        "finite": int(finite.sum()),
        "nonfinite": int((~finite).sum()),
        "nan": int(np.isnan(values).sum()),
        "posinf": int(np.isposinf(values).sum()),
        "neginf": int(np.isneginf(values).sum()),
    }


def _validate_behavior(result):
    behavior = {
        "task_id": int(result["task_id"]),
        "config": str(result["config"]),
        "folds": [
            {
                "fold": fold["fold"],
                "rmse": fold["rmse"],
                "prediction_sha256": fold["prediction_sha256"],
                "metadata": fold["metadata"],
            }
            for fold in result["folds"]
        ],
    }
    if (
        result["behavior_fingerprint_sha256"]
        != runner._json_sha256(behavior)
    ):
        raise RuntimeError("T5 completed worker behavior hash changed")


def build_failure_record(spool_directory):
    registry, _rows = runner._registry()
    expected_task_ids = {
        int(row["task_id"]) for row in registry["tasks"]
    }
    invalid_task_ids = set(INVALID_TARGETS)
    if not invalid_task_ids < expected_task_ids:
        raise RuntimeError("T5 invalid-task declaration changed")
    observed_invalid = {
        task_id: _target_finiteness(task_id)
        for task_id in sorted(invalid_task_ids)
    }
    if observed_invalid != INVALID_TARGETS:
        raise RuntimeError("T5 invalid-target evidence changed")

    expected_completed = expected_task_ids - invalid_task_ids
    paths = sorted(spool_directory.glob(f"task-*--{runner.CONTROL}.json"))
    if len(paths) != len(expected_completed):
        raise RuntimeError("T5 completed-worker count changed")
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        task_id = int(payload["task_id"])
        result, spool_hash = runner._load_spool(
            path,
            payload["binding"],
            task_id,
            runner.CONTROL,
        )
        binding = payload["binding"]
        if (
            binding["runner_sha256"]
            != runner._sha256(Path(runner.__file__).resolve())
            or binding["protocol_sha256"] != runner._sha256(runner.PROTOCOL)
            or binding["darkofit_head"] != EXPECTED_DARKOFIT_HEAD
            or binding["chimeraboost_head"] != runner.EXPECTED_CHIMERA_HEAD
        ):
            raise RuntimeError("T5 completed-worker binding changed")
        _validate_behavior(result)
        records.append(
            {
                "task_id": task_id,
                "config": runner.CONTROL,
                "filename": path.name,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "spool_record_sha256": spool_hash,
                "result_sha256": payload["result_sha256"],
                "behavior_fingerprint_sha256": result[
                    "behavior_fingerprint_sha256"
                ],
            }
        )
    if {row["task_id"] for row in records} != expected_completed:
        raise RuntimeError("T5 completed-worker identities changed")

    if _git_head(ROOT) != EXPECTED_DARKOFIT_HEAD:
        raise RuntimeError("record the T5 failure from its bound commit")
    if _git_head(runner.CHIMERA_ROOT) != runner.EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("T5 ChimeraBoost source changed")
    artifact = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_failure_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "close_t5_composite_candidate",
        "failure_reason": "frozen_panel_contains_nonfinite_targets",
        "campaign_complete": False,
        "outcomes_scored": True,
        "candidate_arm_started": False,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "lockbox_data_used": False,
        "task_drop_allowed": False,
        "task_imputation_allowed": False,
        "rerun_authorized": False,
        "protocol": {
            "path": str(runner.PROTOCOL.relative_to(ROOT)),
            "sha256": runner._sha256(runner.PROTOCOL),
            "runner_path": str(
                Path(runner.__file__).resolve().relative_to(ROOT)
            ),
            "runner_sha256": runner._sha256(
                Path(runner.__file__).resolve()
            ),
            "registry_file_sha256": (
                runner.EXPECTED_REGISTRY_FILE_SHA256
            ),
            "registry_canonical_sha256": (
                runner.EXPECTED_REGISTRY_CANONICAL_SHA256
            ),
        },
        "sources": {
            "darkofit_head": EXPECTED_DARKOFIT_HEAD,
            "chimeraboost_head": runner.EXPECTED_CHIMERA_HEAD,
        },
        "execution": {
            "python": sys.version,
            "interpreter": sys.executable,
            "attempted_wave": runner.CONTROL,
            "expected_worker_count": 25,
            "completed_worker_count": len(records),
            "failed_before_fit_count": len(INVALID_TARGETS),
            "completed_workers": records,
            "invalid_targets": [
                {"task_id": task_id, **observed_invalid[task_id]}
                for task_id in sorted(observed_invalid)
            ],
            "primary_reported_failure_task_id": 362395,
        },
        "panel_disposition": {
            "all_25_lineages_spent_for_confirmation": True,
            "reason": (
                "control outcomes were scored before the frozen data-validity "
                "failure; the panel cannot be repaired or reused"
            ),
        },
    }
    artifact["failure_artifact_sha256"] = runner._json_sha256(artifact)
    return artifact


def _markdown(artifact):
    invalid = artifact["execution"]["invalid_targets"]
    rows = "\n".join(
        f"| {row['task_id']} | {row['dataset_name']} | "
        f"{row['target_name']} | {row['nonfinite']:,} / {row['rows']:,} |"
        for row in invalid
    )
    return f"""# T5 composite confirmation: fail-closed result

**Decision: `close_t5_composite_candidate`.**

The frozen T5 run stopped in its first wave. Twenty-three current-default
workers completed and were persisted; two tasks failed target validation
before fitting. The composite, ChimeraBoost, and CatBoost waves never started.

| Task | Dataset | Target | Non-finite rows |
|---:|---|---|---:|
{rows}

The protocol forbids dropping or imputing a task after outcomes exist. No task
was changed, no run was resumed, and no default promotion is authorized. All
25 lineages are now spent for confirmation because control outcomes were
scored before the failure.

This is a panel-construction failure, not evidence for or against the T5 model
policy. A future nomination needs a new outcome-unseen panel whose target
validity is checked before authorization.
"""


def _atomic_create(path, value):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    creator._atomic_write_bytes(path, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spool-directory",
        type=Path,
        default=DEFAULT_SPOOL_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    args.spool_directory = Path(
        os.path.abspath(os.path.expanduser(args.spool_directory))
    )
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.markdown = Path(os.path.abspath(os.path.expanduser(args.markdown)))
    return args


def main(argv=None):
    args = parse_args(argv)
    artifact = build_failure_record(args.spool_directory)
    _atomic_create(
        args.output,
        (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(),
    )
    _atomic_create(args.markdown, _markdown(artifact).encode())
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
