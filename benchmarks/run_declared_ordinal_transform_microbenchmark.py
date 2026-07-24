#!/usr/bin/env python3
"""Measure the behavior-exact pandas categorical ordinal fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darkofit import DarkoRegressor


SEED = 20260723
SIZES = (128, 4_096, 65_536)
REPEATS = 15
INNER_LOOPS = {128: 50, 4_096: 5, 65_536: 1}
CATEGORIES = {
    "grade": tuple(f"g{index:02d}" for index in range(16)),
    "band": tuple(f"b{index:02d}" for index in range(8)),
    "tier": tuple(f"t{index:02d}" for index in range(4)),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(n_rows: int, *, categorical: bool) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + int(n_rows))
    data: dict[str, object] = {
        f"x{index}": rng.normal(size=n_rows) for index in range(4)
    }
    for offset, (name, categories) in enumerate(CATEGORIES.items()):
        values = np.asarray(categories, dtype=object)[
            rng.integers(0, len(categories), size=n_rows)
        ]
        data[name] = (
            pd.Categorical(values, categories=categories, ordered=True)
            if categorical
            else values
        )
    return pd.DataFrame(data)


def _target(frame: pd.DataFrame) -> np.ndarray:
    result = (
        frame["x0"].to_numpy(dtype=np.float64)
        - 0.4 * frame["x1"].to_numpy(dtype=np.float64)
    )
    for weight, (name, categories) in zip(
        (0.7, -0.3, 0.2), CATEGORIES.items()
    ):
        lookup = {value: index for index, value in enumerate(categories)}
        result = result + weight * np.asarray(
            [lookup[value] for value in frame[name].astype(object)],
            dtype=np.float64,
        )
    return result


def _time_predictions(
    model: DarkoRegressor,
    frame: pd.DataFrame,
    *,
    inner_loops: int,
) -> float:
    started = time.perf_counter_ns()
    for _ in range(inner_loops):
        prediction = model.predict(frame)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise RuntimeError("ordinal prediction benchmark produced invalid output")
    return elapsed / inner_loops


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _run(expected_source_sha: str) -> dict[str, object]:
    source_sha = _git("rev-parse", "HEAD")
    if source_sha != expected_source_sha:
        raise RuntimeError(
            f"source SHA changed: expected {expected_source_sha}, got {source_sha}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("ordinal benchmark requires a clean worktree")

    train = _frame(20_000, categorical=True)
    model = DarkoRegressor(
        iterations=80,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        max_bins=128,
        early_stopping=False,
        linear_leaves=False,
        thread_count=1,
        random_state=SEED,
        diagnostic_warnings="never",
    ).fit(
        train,
        _target(train),
        ordinal_features=CATEGORIES,
    )

    rows: list[dict[str, object]] = []
    for n_rows in SIZES:
        fast = _frame(n_rows, categorical=True)
        generic = fast.astype(
            {
                name: object
                for name in CATEGORIES
            }
        )
        fast_prediction = model.predict(fast)
        generic_prediction = model.predict(generic)
        if not np.array_equal(fast_prediction, generic_prediction):
            raise RuntimeError(
                f"ordinal fast path changed predictions at {n_rows} rows"
            )

        # Warm both routes before collecting paired alternating timings.
        model.predict(fast)
        model.predict(generic)
        fast_seconds: list[float] = []
        generic_seconds: list[float] = []
        for repeat in range(REPEATS):
            ordered = (
                (("fast", fast), ("generic", generic))
                if repeat % 2 == 0
                else (("generic", generic), ("fast", fast))
            )
            for name, frame in ordered:
                seconds = _time_predictions(
                    model,
                    frame,
                    inner_loops=INNER_LOOPS[n_rows],
                )
                (
                    fast_seconds
                    if name == "fast"
                    else generic_seconds
                ).append(seconds)
        ratio = statistics.median(fast_seconds) / statistics.median(
            generic_seconds
        )
        rows.append(
            {
                "n_rows": n_rows,
                "inner_loops": INNER_LOOPS[n_rows],
                "repeats": REPEATS,
                "predictions_bit_exact": True,
                "fast_median_seconds": statistics.median(fast_seconds),
                "generic_median_seconds": statistics.median(generic_seconds),
                "fast_over_generic_ratio": ratio,
                "fast_seconds": fast_seconds,
                "generic_seconds": generic_seconds,
            }
        )

    ratios = [
        float(row["fast_over_generic_ratio"])
        for row in rows
    ]
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "seed": SEED,
        "sizes": list(SIZES),
        "category_cardinalities": {
            name: len(categories) for name, categories in CATEGORIES.items()
        },
        "behavior_exact": all(
            bool(row["predictions_bit_exact"]) for row in rows
        ),
        "all_shapes_faster": all(ratio < 1.0 for ratio in ratios),
        "equal_shape_geomean_ratio": _geometric_mean(ratios),
        "rows": rows,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def _render(result: dict[str, object], raw_sha256: str) -> str:
    rows = result["rows"]
    lines = [
        "# Declared-ordinal transform microbenchmark",
        "",
        "This development microbenchmark compares the direct pandas categorical",
        "code path with the behavior-equivalent generic object-mapping path.",
        "It is a mechanism check, not general quality or hardware evidence.",
        "",
        f"- Source: `{result['source_sha']}`",
        f"- Raw SHA-256: `{raw_sha256}`",
        f"- Predictions bit-exact: `{str(result['behavior_exact']).lower()}`",
        (
            "- Equal-shape geomean fast/generic prediction ratio: "
            f"`{result['equal_shape_geomean_ratio']:.6f}`"
        ),
        f"- Faster at every measured shape: `{str(result['all_shapes_faster']).lower()}`",
        "",
        "| Rows | Fast seconds | Generic seconds | Fast / generic |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_rows']} | {row['fast_median_seconds']:.9f} | "
            f"{row['generic_median_seconds']:.9f} | "
            f"{row['fast_over_generic_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The generic route is retained as the correctness fallback whenever",
            "the incoming categorical dtype does not exactly match the fitted",
            "declared category order.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.raw_output, args.result_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    result = _run(args.expected_source_sha)
    raw_bytes = (
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with args.raw_output.open("xb") as handle:
        handle.write(raw_bytes)
    raw_sha256 = _sha256(args.raw_output)
    with args.result_output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_render(result, raw_sha256))


if __name__ == "__main__":
    main()
