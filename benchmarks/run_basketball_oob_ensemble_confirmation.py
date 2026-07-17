#!/usr/bin/env python3
"""Run the frozen basketball OOB-ensemble stable confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_basketball_oob_ensemble as original  # noqa: E402


EXPECTED_THREADS = 18
TIMING_BLOCKS = 6
MAX_ARM_IQR_FRACTION = 0.20
MAX_RATIO_IQR_FRACTION = 0.15
EXPECTED_PACKAGE_MANIFEST = (
    "4fe4830c9c36de36ce29e626743c83fa0f20e60db8f15e2c7ccd5c96d3226068"
)
EXPECTED_PROTOCOL_SHA256 = (
    "a8fd26868471028fb9e652fee6153e2a7ba9f57fe558407f07d890d8917903bf"
)
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
    "benchmarks/run_basketball_oob_ensemble.py": (
        "dc04a3ee6a61e0f8bf13244543968abb3f3db589c2cb183c128244160dccd631"
    ),
    "benchmarks/basketball_oob_ensemble.json": (
        "1ecfb8dfa2e3e756bd5eae2dfc47b8da4c7c21ea92c875f7dff8ad6b4ddb25bd"
    ),
}
PROTOCOL_PATH = ROOT / "benchmarks/basketball_oob_ensemble_confirmation_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks/basketball_oob_ensemble_confirmation.json"

EXPECTED_GOLDENS = {
    original.DEFAULT_CONFIG: {
        "mean_r2": 0.5267495183883605,
        "fold_prediction_sha256": [
            "6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f",
            "96ad500c63ac3701fe769b03a369d3a01ed1af9695d71c7ea68936d36479da44",
            "230b3cb530dee9ba8f5196b2b12b77f8d62751c545828ca13bad3fe04e54261b",
            "4603c6b3036bbdee060faaa92e6eee18a1f803e4abe9bc4aa7906745db5bd1c1",
            "e00b84d4aa7b8640aad72f5aed6e5e578cef2035459aa146b972145dc8d19fef",
            "12852587a9d1cd729cde1b28d714ff0c30b8051e806d6bb2f3f68088f22912d8",
            "514663b32f0adaf0fc7591def75632f5ea1103598b2d7aaeeaf37fdc2560bb04",
            "45374906a6931f90a6fff29ba0544c4d66311bb6152e3f250d54db55e0c03384",
            "32167d2ad1ba4ee34297a812be85ae67675f638383d61fe709b130bdbb3931a5",
            "f51972e8f896568291b259d698726b224a2399711f8e8cdf451e68b5090ae38d",
        ],
        "holdout_prediction_sha256": (
            "5d910ae8f6b0dca563b99f9f881dcb17ee092711a46b2890452eaa3b8e68367a"
        ),
        "cold_prediction_sha256": (
            "998a14f530ed284865a50726191da067f72d69da3001614d664a4b90e7aa6376"
        ),
        "seen_prediction_sha256": (
            "c9b506afbfb3eb660dd918ee9635d996c0285b0320ba250cbf39c80df9122425"
        ),
    },
    original.CANDIDATE_CONFIG: {
        "mean_r2": 0.5306250399874948,
        "fold_prediction_sha256": [
            "602eb96a14d4b6cfb2cbb88ed1e8d53b05d6b54ff7d34c460630dd2ff7f77186",
            "159b8708d4ab92657d08516fadd2ed3aea689a95a5d7e29c5945c62b25edc6b8",
            "a6416b926ea843fc678b2a8f884e8a3b682a4bbddae9137b10c699a3f6d0417b",
            "03b1d3c4903c07ab37065d76c8dff17ecccb9fa07d0252b2ba46d92a2f4d87c3",
            "8800578d8d36bba655b280186a56bccd13999d79c8ce0030acce43c0d72f8473",
            "bc696811bea3c5900532c056658d74a985849f2328fda362f23fdcbe596f94d2",
            "c754a201b4eaa7628d7653378304a4573c046c7325a67fd99c317daa37b7c29e",
            "af8df78760239f76d69ca351395e535e007a4a745203d3b13a23d4b9919fbe71",
            "1eab18aa9c4f1b3023ad16f25970589cc492eb2ab7e4205e9958b6a62b533617",
            "a10b3812ee6c06bc092ff0361b68fd5eb0b83339690b633424a1cf6caf657abe",
        ],
        "holdout_prediction_sha256": (
            "93dd2d6dbb0d7b41e9c02da7256277ccb385960b1a19813c7405d0526302b3c4"
        ),
        "cold_prediction_sha256": (
            "9de5f0d47a4b6cd4d44693dcea2bca876eddec47947276d58ad50eda3d65847a"
        ),
        "seen_prediction_sha256": (
            "e1f17f1eb6f5684ffb9afbdb40a12c38b047ee8c6e8ca3c6ea35f7c0098b1799"
        ),
    },
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_content_manifest(repo: Path, prefix: str) -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", prefix]
    )
    paths = sorted(item.decode() for item in raw.split(b"\0") if item)
    digest = hashlib.sha256()
    for relative in paths:
        name = relative.encode()
        content = (repo / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def _source_state() -> dict[str, Any]:
    return {
        "repository": str(ROOT),
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "origin_main": _git("rev-parse", "origin/main"),
        "status_porcelain": _git(
            "status", "--porcelain", "--untracked-files=all"
        ),
        "package_manifest_sha256": _tracked_content_manifest(ROOT, "darkofit"),
        "support_sha256": {
            name: _sha256_file(ROOT / name) for name in EXPECTED_SUPPORT_SHA256
        },
    }


def require_clean_frozen_source() -> dict[str, Any]:
    state = _source_state()
    if state["branch"] != "main":
        raise RuntimeError("formal OOB confirmation requires main")
    if state["head"] != state["origin_main"]:
        raise RuntimeError("formal OOB confirmation requires pushed main")
    if state["status_porcelain"]:
        raise RuntimeError("formal OOB confirmation requires clean source")
    if state["package_manifest_sha256"] != EXPECTED_PACKAGE_MANIFEST:
        raise RuntimeError("DarkoFit package manifest changed")
    if state["support_sha256"] != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("OOB confirmation support files changed")
    if _sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("OOB confirmation protocol changed")
    return state


def schedule() -> tuple[tuple[str, str], ...]:
    return harness.reciprocal_schedule(
        original.DEFAULT_CONFIG,
        original.CANDIDATE_CONFIG,
        repetitions=TIMING_BLOCKS,
    )


def _worker_result(stdout: str) -> tuple[dict[str, Any], str | None]:
    lines = stdout.splitlines()
    matches = [
        line for line in lines if line.startswith(original.WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("OOB worker did not emit exactly one result")
    payload = json.loads(matches[0][len(original.WORKER_RESULT_PREFIX) :])
    chatter = "\n".join(
        line for line in lines if not line.startswith(original.WORKER_RESULT_PREFIX)
    ).strip()
    return payload, chatter or None


def run_worker_process(
    config: str, *, threads: int, data_cache: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "benchmarks/run_basketball_oob_ensemble.py"),
        "--worker-config",
        config,
        "--threads",
        str(threads),
        "--data-cache",
        str(data_cache),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=harness.worker_environment(threads),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"OOB worker {config!r} failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result, chatter = _worker_result(completed.stdout)
    result["worker_stdout"] = chatter
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def validate_golden(result: dict[str, Any]) -> None:
    config = result.get("config")
    if config not in EXPECTED_GOLDENS:
        raise RuntimeError("unknown OOB confirmation arm")
    expected = EXPECTED_GOLDENS[config]
    if not math.isclose(
        float(result["mean_r2"]),
        float(expected["mean_r2"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"{config} mean basketball R2 changed")
    fold_hashes = [fold["prediction_sha256"] for fold in result["folds"]]
    if fold_hashes != expected["fold_prediction_sha256"]:
        raise RuntimeError(f"{config} fold prediction goldens changed")
    if result["holdout"]["prediction_sha256"] != expected[
        "holdout_prediction_sha256"
    ]:
        raise RuntimeError(f"{config} held-team prediction golden changed")
    scores = result["holdout"]["scores"]
    if scores["cold_player_subset"]["prediction_sha256"] != expected[
        "cold_prediction_sha256"
    ]:
        raise RuntimeError(f"{config} cold-player prediction golden changed")
    if scores["seen_player_subset"]["prediction_sha256"] != expected[
        "seen_prediction_sha256"
    ]:
        raise RuntimeError(f"{config} seen-player prediction golden changed")


def timing_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.shape != (TIMING_BLOCKS,):
        raise RuntimeError(f"timing requires exactly {TIMING_BLOCKS} values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RuntimeError("timing values must be positive and finite")
    median = float(np.median(array))
    iqr = float(np.subtract(*np.percentile(array, [75, 25])))
    return {
        "values_seconds": [float(value) for value in array],
        "minimum_seconds": float(array.min()),
        "median_seconds": median,
        "maximum_seconds": float(array.max()),
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median,
        "maximum_over_minimum": float(array.max() / array.min()),
    }


def quality_summary(canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    default = canonical[original.DEFAULT_CONFIG]
    candidate = canonical[original.CANDIDATE_CONFIG]
    default_scores = np.asarray(default["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    deltas = candidate_scores - default_scores
    jackknife = [
        float(np.mean(np.delete(deltas, fold))) for fold in range(len(deltas))
    ]
    default_holdout = default["holdout"]["scores"]
    candidate_holdout = candidate["holdout"]["scores"]
    team_delta = float(
        candidate_holdout["overlap_exposed_team_holdout"]["r2"]
        - default_holdout["overlap_exposed_team_holdout"]["r2"]
    )
    cold_delta = float(
        candidate_holdout["cold_player_subset"]["r2"]
        - default_holdout["cold_player_subset"]["r2"]
    )
    gates = {
        "mean_r2_no_regression": float(np.mean(deltas)) >= 0.0,
        "fold_breadth": int(np.count_nonzero(deltas > 0.0)) >= 6,
        "leave_one_fold_out_no_regression": min(jackknife) >= 0.0,
        "overlap_exposed_team_no_regression": team_delta >= 0.0,
        "cold_player_no_regression": cold_delta >= 0.0,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "mean_r2_delta": float(np.mean(deltas)),
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "leave_one_fold_out_mean_deltas": jackknife,
        "overlap_exposed_team_r2_delta": team_delta,
        "cold_player_r2_delta": cold_delta,
    }


def analyze(
    canonical: dict[str, dict[str, Any]],
    behavior_fingerprints: dict[str, set[str]],
    wall: dict[str, dict[str, Any]],
    prediction: dict[str, dict[str, Any]],
    paired_ratio: dict[str, Any],
) -> dict[str, Any]:
    quality = quality_summary(canonical)
    default = original.DEFAULT_CONFIG
    candidate = original.CANDIDATE_CONFIG
    wall_ratio = wall[candidate]["median_seconds"] / wall[default]["median_seconds"]
    prediction_ratio = (
        prediction[candidate]["median_seconds"]
        / prediction[default]["median_seconds"]
    )
    gates = {
        "quality_reproduced": quality["passed"],
        "behavior_repeat_exact": all(
            len(values) == 1 for values in behavior_fingerprints.values()
        ),
        "wall_timing_stable": all(
            value["iqr_fraction"] <= MAX_ARM_IQR_FRACTION
            for value in wall.values()
        ),
        "prediction_timing_stable": all(
            value["iqr_fraction"] <= MAX_ARM_IQR_FRACTION
            for value in prediction.values()
        ),
        "paired_wall_ratio_stable": (
            paired_ratio["iqr_fraction"] <= MAX_RATIO_IQR_FRACTION
        ),
        "beats_naive_fivefold_wall_scaling": (
            wall_ratio <= original.MAX_TOTAL_RUNTIME_RATIO
        ),
        "prediction_cost_within_budget": (
            prediction_ratio <= original.MAX_PREDICT_RUNTIME_RATIO
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "quality": quality,
        "candidate_over_default_median_wall_time": wall_ratio,
        "candidate_over_default_median_predict_time": prediction_ratio,
        "passed": passed,
        "candidate_scope": "opt_in_only",
        "default_promotion_authorized": False,
        "recommendation": (
            "advance_to_opt_in_api_implementation"
            if passed
            else "close_oob_ensemble_attempt"
        ),
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads != EXPECTED_THREADS:
        raise ValueError(f"confirmation requires {EXPECTED_THREADS} threads")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    # Prime and validate the immutable dataset in the parent.  Every timed
    # worker must therefore observe the same ``load_source == \"cache\"``
    # metadata that participates in the frozen behavior fingerprint.
    harness.load_basketball_dataset(args.data_cache)
    source = require_clean_frozen_source()
    run_schedule = schedule()
    canonical: dict[str, dict[str, Any]] = {}
    fingerprints = {name: set() for name in original.CONFIG_ORDER}
    wall_values = {name: [] for name in original.CONFIG_ORDER}
    prediction_values = {name: [] for name in original.CONFIG_ORDER}
    repeats = []
    paired_values = []
    for block, order in enumerate(run_schedule):
        block_wall = {}
        for position, config in enumerate(order):
            current = _source_state()
            if current != source:
                raise RuntimeError("DarkoFit source changed during confirmation")
            print(
                f"block {block + 1}/{TIMING_BLOCKS}, "
                f"position {position + 1}: {config}",
                flush=True,
            )
            load_before = [float(value) for value in os.getloadavg()]
            result = run_worker_process(
                config, threads=args.threads, data_cache=args.data_cache
            )
            load_after = [float(value) for value in os.getloadavg()]
            validate_golden(result)
            fingerprint = result["behavior_fingerprint_sha256"]
            fingerprints[config].add(fingerprint)
            wall = float(result["steady_wall_seconds"])
            predict = float(result["summed_predict_seconds"])
            wall_values[config].append(wall)
            prediction_values[config].append(predict)
            block_wall[config] = wall
            canonical.setdefault(config, result)
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "config": config,
                    "steady_wall_seconds": wall,
                    "summed_predict_seconds": predict,
                    "summed_fit_seconds": result["summed_fit_seconds"],
                    "warmup_seconds_outside_timing": result[
                        "warmup_seconds_outside_timing"
                    ],
                    "behavior_fingerprint_sha256": fingerprint,
                    "load_average_before": load_before,
                    "load_average_after": load_after,
                    "worker_stdout": result["worker_stdout"],
                    "worker_stderr": result["worker_stderr"],
                }
            )
            print(
                f"  mean R2={result['mean_r2']:.12f}; steady={wall:.3f}s",
                flush=True,
            )
        paired_values.append(
            block_wall[original.CANDIDATE_CONFIG]
            / block_wall[original.DEFAULT_CONFIG]
        )
    if _source_state() != source:
        raise RuntimeError("DarkoFit source changed during confirmation")
    wall_summary = {
        name: timing_summary(values) for name, values in wall_values.items()
    }
    prediction_summary = {
        name: timing_summary(values)
        for name, values in prediction_values.items()
    }
    paired_summary = timing_summary(paired_values)
    decision = analyze(
        canonical,
        fingerprints,
        wall_summary,
        prediction_summary,
        paired_summary,
    )
    payload = {
        "schema_version": 1,
        "campaign": "basketball_oob_ensemble_stable_confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "primary_dataset": "basketball",
            "cold_player_guardrail": True,
            "ctr23_used": False,
            "threads": EXPECTED_THREADS,
            "timing_blocks": TIMING_BLOCKS,
            "schedule": [list(order) for order in run_schedule],
            "max_arm_iqr_fraction": MAX_ARM_IQR_FRACTION,
            "max_paired_ratio_iqr_fraction": MAX_RATIO_IQR_FRACTION,
        },
        "runner_sha256": _sha256_file(Path(__file__)),
        "canonical_results": [
            canonical[name] for name in original.CONFIG_ORDER
        ],
        "timing_repeats": repeats,
        "wall_timing": wall_summary,
        "prediction_timing": prediction_summary,
        "paired_wall_ratio": paired_summary,
        "behavior_fingerprints": {
            name: sorted(values) for name, values in fingerprints.items()
        },
        "decision": decision,
    }
    _write_create_only(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    # Use lexical absolute paths here.  ``Path.resolve`` follows symlinks and
    # would hide them from the explicit refusal checks in ``run`` and the data
    # loader.
    args.data_cache = Path(os.path.abspath(args.data_cache.expanduser()))
    args.output = Path(os.path.abspath(args.output.expanduser()))
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
