#!/usr/bin/env python3
"""Run selector-v3 on the three spent Protein development coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
BENCH = RUNNER_PATH.parent
PROTOCOL_PATH = BENCH / "automatic_linear_selector_v3_protein.md"
TEST_PATH = ROOT / "tests/test_automatic_linear_selector_v3_protein.py"
CONTRACT_ID = "automatic-linear-selector-v3-protein-development-20260723"
MECHANISM_ID = "automatic_linear_selector_v3"

from benchmarks import run_automatic_linear_selector_v2_protein_attribution as base
from benchmarks import (
    run_automatic_linear_selector_v2_protein_attribution_attempt2 as loader,
)


def _validate_clean_source(path: Path):
    state = base.source_state(path)
    if not state["clean"]:
        raise RuntimeError("selector-v3 Protein run requires clean source")
    return state


def _validate_harness():
    state = _validate_clean_source(ROOT)
    for path in (PROTOCOL_PATH, RUNNER_PATH, TEST_PATH):
        if not path.is_file() or path.read_bytes() != base._tracked_head_bytes(path):
            raise RuntimeError(f"bound harness file differs from HEAD: {path.name}")
    return state


def _validate_candidate(path: Path):
    state = _validate_clean_source(path)
    if state["head"] != base._git(ROOT, "rev-parse", "HEAD"):
        raise RuntimeError("candidate must be the clean harness source")
    return state


def _validate_bound_evidence():
    paths = (
        BENCH / "automatic_linear_selector_v3_noise_calibration.md",
        BENCH / "automatic_linear_selector_v3_noise_calibration_result.md",
        BENCH / "automatic_linear_selector_v3_noise_calibration_20260723.json",
        BENCH / "automatic_linear_selector_v3_noise_calibration_2se_20260723.json",
    )
    return {
        str(path.relative_to(ROOT)): {"sha256": base.sha256(path)}
        for path in paths
    }


def _geomean(values):
    return float(math.exp(sum(math.log(float(value)) for value in values) / len(values)))


def analyze_rows(rows):
    expected = base.expected_ordered_grid()
    if len(rows) != len(expected):
        raise RuntimeError("selector-v3 Protein grid is incomplete")
    by_key = {}
    for row, cell in zip(rows, expected):
        identity = {
            name: row.get(name)
            for name in ("coordinate", "repeat", "fold", "seed", "position", "arm")
        }
        if identity != cell:
            raise RuntimeError("selector-v3 Protein row order drifted")
        key = (int(row["coordinate"]), str(row["arm"]))
        if key in by_key:
            raise RuntimeError("selector-v3 Protein row is duplicated")
        by_key[key] = row

    coordinate_results = []
    ratios = []
    exact_selected_coordinates = 0
    for coordinate in base.COORDINATES:
        index = int(coordinate["coordinate"])
        constant = by_key[(index, "constant")]
        automatic = by_key[(index, "automatic")]
        explicit = by_key[(index, "explicit_linear")]
        selector = automatic.get("selector")
        if (
            constant["fingerprints"]
            != automatic["fingerprints"]
            or automatic["fingerprints"] != explicit["fingerprints"]
            or not isinstance(selector, dict)
            or selector.get("version") != 2
            or selector.get("eligible") is not True
            or selector.get("minimum_gain_z") != 2.0
        ):
            raise RuntimeError("selector-v3 Protein provenance is invalid")
        for row in (constant, automatic, explicit):
            if (
                row.get("environment") != base.WORKER_ENVIRONMENT
                or row.get("warnings") is None
                or row.get("fit_rss", {}).get("errors") != []
                or any(
                    row.get(name) != base.THREADS
                    for name in (
                        "numba_threads_before_fit",
                        "numba_threads_after_fit",
                        "numba_threads_after_predict",
                        "numba_threads_after_timing",
                    )
                )
            ):
                raise RuntimeError("selector-v3 Protein worker state is invalid")
        ratio = float(automatic["test_rmse"] / constant["test_rmse"])
        ratios.append(ratio)
        selected = selector.get("resolved_linear_leaves") is True
        exact_to_selected = (
            automatic["prediction_sha256"] == explicit["prediction_sha256"]
            and automatic["core_booster_state_sha256"]
            == explicit["core_booster_state_sha256"]
            if selected
            else automatic["prediction_sha256"] == constant["prediction_sha256"]
            and automatic["core_booster_state_sha256"]
            == constant["core_booster_state_sha256"]
        )
        if selected and exact_to_selected:
            exact_selected_coordinates += 1
        coordinate_results.append({
            "coordinate": index,
            "automatic_over_constant_rmse": ratio,
            "constant_rmse": float(constant["test_rmse"]),
            "automatic_rmse": float(automatic["test_rmse"]),
            "explicit_linear_rmse": float(explicit["test_rmse"]),
            "selected_linear": selected,
            "selection_reason": selector["reason"],
            "relative_validation_improvement": float(
                selector["relative_validation_improvement"]
            ),
            "paired_mse_gain_z": float(selector["paired_mse_gain_z"]),
            "exact_to_resolved_arm": exact_to_selected,
        })
    aggregate = _geomean(ratios)
    worst = max(ratios)
    exact = all(item["exact_to_resolved_arm"] for item in coordinate_results)
    clearly_better = aggregate < 1.0 and worst <= 1.0 and exact
    return {
        "disposition": (
            "ready_for_holdout_ship_check"
            if clearly_better
            else "keep_opt_in"
        ),
        "aggregate_automatic_over_constant_rmse": aggregate,
        "worst_coordinate_ratio": worst,
        "coordinate_count": len(coordinate_results),
        "selected_coordinate_count": exact_selected_coordinates,
        "all_resolved_arms_exact": exact,
        "clearly_better_without_observed_coordinate_harm": clearly_better,
        "coordinates": coordinate_results,
    }


@contextmanager
def configured_base():
    patches = {
        "RUNNER_PATH": RUNNER_PATH,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "TEST_PATH": TEST_PATH,
        "CONTRACT_ID": CONTRACT_ID,
        "MECHANISM_ID": MECHANISM_ID,
        "ATTEMPT_INDEX": 1,
        "CANDIDATE_COMMIT": base._git(ROOT, "rev-parse", "HEAD"),
        "EXPECTED_HASHES": {},
        "_validate_harness": _validate_harness,
        "_validate_candidate": _validate_candidate,
        "validate_bound_evidence": _validate_bound_evidence,
        "_load_split": loader._load_split,
        "_data_loader_preflight": loader._data_loader_preflight,
        "analyze_rows": analyze_rows,
    }
    originals = {name: getattr(base, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--tabarena-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-started-at", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    worker_fields = (
        args.worker_index,
        args.arm,
        args.parent_pid,
        args.worker_started_at,
    )
    if any(value is not None for value in worker_fields) and not all(
        value is not None for value in worker_fields
    ):
        parser.error("internal worker arguments must be supplied together")
    if args.worker_index is not None and (
        args.worker_index not in range(len(base.expected_ordered_grid()))
        or args.arm not in base.ARMS
    ):
        parser.error("worker identity is invalid")
    return args


def _worker_command(args, worker_index, arm):
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--candidate-source",
        str(args.candidate_source.resolve()),
        "--tabarena-source",
        str(args.tabarena_source.resolve()),
        "--output",
        str(args.output.resolve()),
        "--worker-index",
        str(worker_index),
        "--arm",
        arm,
        "--parent-pid",
        str(os.getpid()),
        "--worker-started-at",
        datetime.now(timezone.utc).isoformat(),
    ]


def run_parent(args):
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    before = {
        "harness": _validate_harness(),
        "candidate": _validate_candidate(args.candidate_source),
        "tabarena": base.validate_tabarena_source(args.tabarena_source),
    }
    preflight = loader._data_loader_preflight(
        args.candidate_source, args.tabarena_source
    )
    rows = []
    for worker_index, cell in enumerate(base.expected_ordered_grid()):
        completed = subprocess.run(
            _worker_command(args, worker_index, cell["arm"]),
            cwd=ROOT,
            env={**os.environ, **base.WORKER_ENVIRONMENT},
            check=False,
            capture_output=True,
            text=True,
            timeout=base.WORKER_TIMEOUT_SECONDS,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Protein worker {worker_index} failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        row = base._parse_worker_stdout(completed.stdout)
        rows.append(row)
        print(
            f"ok {worker_index + 1}/{len(base.expected_ordered_grid())} "
            f"coordinate={cell['coordinate']} arm={cell['arm']}",
            flush=True,
        )
    after = {
        "harness": _validate_harness(),
        "candidate": _validate_candidate(args.candidate_source),
        "tabarena": base.validate_tabarena_source(args.tabarena_source),
    }
    if after != before:
        raise RuntimeError("selector-v3 Protein sources changed during execution")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": CONTRACT_ID,
        "mechanism_id": MECHANISM_ID,
        "evidence_scope": "spent_protein_development",
        "sources": before,
        "bindings": _validate_bound_evidence(),
        "data_loader_preflight": preflight,
        "grid": base.expected_ordered_grid(),
        "rows": rows,
        "analysis": analyze_rows(rows),
        "limitations": [
            "spent development coordinates",
            "not holdout evidence",
            "not a shipping claim",
        ],
    }
    base._write_create_only_json(output, payload)
    print(f"wrote selector-v3 Protein result to {output}")
    print(f"artifact sha256: {base.sha256(output)}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    with configured_base():
        if args.worker_index is not None:
            result = base._worker_result(args)
            print(
                base.WORKER_PREFIX
                + json.dumps(result, allow_nan=False, sort_keys=True)
            )
            return 0
        return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
