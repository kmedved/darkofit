"""WNBA DARKO observation-noise shadow replay with ChimeraBoost variance.

This is a research benchmark for the Kalman-readiness gate. It uses the
game-level WNBA DARKO metric observation artifact, fits the calibrated Gaussian
distributional head on train/validation seasons, then replays a scalar
random-walk Kalman filter per metric over held-out seasons with either:

* incumbent heuristic observation variance: ``R_t = sigma2 / sample_weight``;
* ChimeraBoost observation variance: ``R_t = predict_variance(X_t)``.

The production DARKO player filter is not modified here; this script is a
reproducible shadow replay of the observation-noise contract on the artifact
that motivated the calibration work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench_wnba_realdata_distributional import DEFAULT_DATA, load_dataset
from chimeraboost import ChimeraBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "benchmarks" / "wnba_kalman_replay.csv"
DEFAULT_SUMMARY = ROOT / "benchmarks" / "wnba_kalman_replay_summary.md"
Z90 = 1.6448536269514722


@dataclass
class ReplayResult:
    model: str
    metric: str
    season: str
    n: int
    weight_sum: float
    nll: float
    rmse: float
    mae: float
    coverage90: float
    nis_mean: float
    std_innov_rms: float
    std_innov_mean: float
    lag1_std_innov_corr: float
    mean_R: float
    median_R: float
    q: float
    r_scale: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-through-season", type=int, default=2022)
    parser.add_argument("--val-start-season", type=int, default=2023)
    parser.add_argument("--val-through-season", type=int, default=2023)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=25)
    parser.add_argument("--l2-leaf-reg", default="auto")
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--r-floor", type=float, default=0.01)
    parser.add_argument("--r-ceil", type=float, default=9.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    df, X, y, w, feature_cols = load_dataset(args.data)
    train_mask = df["season"] <= args.train_through_season
    val_mask = (
        (df["season"] >= args.val_start_season)
        & (df["season"] <= args.val_through_season)
    )
    test_mask = df["season"] > args.val_through_season
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("empty train/validation/test split")

    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping=True,
        early_stopping_rounds=args.early_stopping_rounds,
        dist_calibration="per_metric_affine",
        dist_calibration_feature=0,
        eval_metric="nll",
        random_state=args.random_state,
        thread_count=args.thread_count,
        diagnostic_warnings="never",
    )
    model.fit(
        X[train_mask],
        y[train_mask],
        cat_features=[0],
        eval_set=(X[val_mask], y[val_mask]),
        sample_weight=w[train_mask],
        eval_sample_weight=w[val_mask],
    )
    chimera_R = np.clip(
        model.predict_variance(X),
        float(args.r_floor),
        float(args.r_ceil),
    )

    rows: list[ReplayResult] = []
    details = {
        "best_iteration": int(model.best_n_estimators_),
        "sigma_affine_b": float(getattr(model, "sigma_affine_b_", np.nan)),
        "group_count": int(len(getattr(model, "dist_group_affine_metadata_", ()))),
    }
    for metric in sorted(df["metric"].unique()):
        metric_mask = df["metric"].eq(metric).to_numpy()
        idx = np.flatnonzero(metric_mask)
        if idx.size == 0:
            continue
        y_m = y[idx]
        w_m = w[idx]
        dates_m = df.loc[metric_mask, "date"].to_numpy()
        game_m = df.loc[metric_mask, "game_id"].to_numpy()
        seasons_m = df.loc[metric_mask, "season"].to_numpy(dtype=np.int64)
        train_m = train_mask.to_numpy()[idx]
        val_m = val_mask.to_numpy()[idx]
        test_m = test_mask.to_numpy()[idx]
        q_chimera = tune_q(y_m, w_m, dates_m, game_m, chimera_R[idx], train_m, val_m)
        heuristic_R, q_heuristic, r_scale = tune_heuristic_r(
            y_m, w_m, dates_m, game_m, train_m, val_m
        )
        metric_results = [
            (
                "chimera_variance",
                chimera_R[idx],
                q_chimera,
                None,
            ),
            (
                "incumbent_weight_heuristic",
                heuristic_R,
                q_heuristic,
                r_scale,
            ),
        ]
        for name, R_m, q, r_scale_value in metric_results:
            replay = run_replay(y_m, w_m, dates_m, game_m, R_m, q)
            rows.extend(score_replay(name, metric, replay, test_m, seasons_m, q, r_scale_value))

    rows.extend(overall_rows(rows))
    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary, args, details, time.perf_counter() - start)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_summary}")
    for row in rows:
        if row.metric == "ALL" and row.season == "ALL":
            print(
                row.model,
                "nll",
                f"{row.nll:.3f}",
                "rmse",
                f"{row.rmse:.3f}",
                "coverage90",
                f"{row.coverage90:.3f}",
                "nis",
                f"{row.nis_mean:.3f}",
            )


def tune_q(y, w, dates, game_id, R, train_mask, val_mask):
    best_q = None
    best_nll = float("inf")
    for q in np.logspace(-6.0, 0.5, 32):
        replay = run_replay(y, w, dates, game_id, R, float(q))
        row = score_arrays(replay, val_mask)
        if row["nll"] < best_nll:
            best_nll = row["nll"]
            best_q = float(q)
    return float(best_q if best_q is not None else 0.0)


def tune_heuristic_r(y, w, dates, game_id, train_mask, val_mask):
    train_y = y[train_mask]
    train_w = w[train_mask]
    center = weighted_mean(train_y, train_w)
    base_var = weighted_mean((train_y - center) ** 2, train_w)
    base_sigma2 = max(base_var * weighted_mean(train_w, train_w), 1.0e-6)
    best = None
    for scale in np.logspace(-1.0, 1.0, 25):
        R = np.maximum(base_sigma2 * float(scale) / np.maximum(w, 1.0e-6), 1.0e-6)
        q = tune_q(y, w, dates, game_id, R, train_mask, val_mask)
        replay = run_replay(y, w, dates, game_id, R, q)
        row = score_arrays(replay, val_mask)
        key = (row["nll"], row["rmse"])
        if best is None or key < best["key"]:
            best = {"key": key, "R": R, "q": q, "scale": float(scale)}
    return best["R"], float(best["q"]), float(best["scale"])


def run_replay(y, w, dates, game_id, R, q):
    order = np.lexsort((game_id, dates))
    y_ord = y[order]
    w_ord = w[order]
    R_ord = np.maximum(R[order], 1.0e-9)
    date_ord = dates[order]
    mean = weighted_mean(y_ord, w_ord)
    var = max(weighted_mean((y_ord - mean) ** 2, w_ord), 1.0e-6)
    prev_date = None
    pred = np.empty_like(y_ord, dtype=np.float64)
    pred_var = np.empty_like(y_ord, dtype=np.float64)
    obs_var = np.empty_like(y_ord, dtype=np.float64)
    innovation = np.empty_like(y_ord, dtype=np.float64)
    standardized = np.empty_like(y_ord, dtype=np.float64)
    for pos, idx in enumerate(order):
        current_date = date_ord[pos]
        if prev_date is not None:
            days = max(
                0.0,
                (np.datetime64(current_date, "D") - np.datetime64(prev_date, "D"))
                / np.timedelta64(1, "D"),
            )
            var += float(q) * days
        prev_date = current_date
        S = max(var + R_ord[pos], 1.0e-9)
        err = y_ord[pos] - mean
        pred[pos] = mean
        pred_var[pos] = S
        obs_var[pos] = R_ord[pos]
        innovation[pos] = err
        standardized[pos] = err / math.sqrt(S)
        gain = var / S
        mean = mean + gain * err
        var = max((1.0 - gain) * var, 1.0e-9)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return {
        "pred": pred[inv],
        "pred_var": pred_var[inv],
        "obs_var": obs_var[inv],
        "innovation": innovation[inv],
        "standardized": standardized[inv],
    }


def score_replay(model, metric, replay, test_mask, seasons, q, r_scale):
    rows = []
    rows.append(make_result(model, metric, "ALL", replay, test_mask, q, r_scale))
    for season in sorted(np.unique(seasons[test_mask])):
        mask = test_mask & (seasons == season)
        rows.append(make_result(model, metric, str(int(season)), replay, mask, q, r_scale))
    return rows


def make_result(model, metric, season, replay, mask, q, r_scale):
    scored = score_arrays(replay, mask)
    return ReplayResult(
        model=model,
        metric=metric,
        season=season,
        n=int(np.sum(mask)),
        weight_sum=float(scored["weight_sum"]),
        nll=float(scored["nll"]),
        rmse=float(scored["rmse"]),
        mae=float(scored["mae"]),
        coverage90=float(scored["coverage90"]),
        nis_mean=float(scored["nis_mean"]),
        std_innov_rms=float(scored["std_innov_rms"]),
        std_innov_mean=float(scored["std_innov_mean"]),
        lag1_std_innov_corr=float(scored["lag1_std_innov_corr"]),
        mean_R=float(scored["mean_R"]),
        median_R=float(scored["median_R"]),
        q=float(q),
        r_scale=None if r_scale is None else float(r_scale),
    )


def score_arrays(replay, mask):
    mask = np.asarray(mask, dtype=bool)
    innovation = replay["innovation"][mask]
    pred_var = np.maximum(replay["pred_var"][mask], 1.0e-9)
    z = replay["standardized"][mask]
    R = replay["obs_var"][mask]
    if innovation.size == 0:
        return {
            "weight_sum": 0.0,
            "nll": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "coverage90": float("nan"),
            "nis_mean": float("nan"),
            "std_innov_rms": float("nan"),
            "std_innov_mean": float("nan"),
            "lag1_std_innov_corr": float("nan"),
            "mean_R": float("nan"),
            "median_R": float("nan"),
        }
    weights = np.ones_like(innovation, dtype=np.float64)
    nll = 0.5 * np.log(2.0 * np.pi * pred_var) + 0.5 * innovation * innovation / pred_var
    return {
        "weight_sum": float(np.sum(weights)),
        "nll": weighted_mean(nll, weights),
        "rmse": math.sqrt(weighted_mean(innovation * innovation, weights)),
        "mae": weighted_mean(np.abs(innovation), weights),
        "coverage90": weighted_mean((np.abs(z) <= Z90).astype(np.float64), weights),
        "nis_mean": weighted_mean(z * z, weights),
        "std_innov_rms": math.sqrt(weighted_mean(z * z, weights)),
        "std_innov_mean": weighted_mean(z, weights),
        "lag1_std_innov_corr": lag1_corr(z),
        "mean_R": float(np.mean(R)),
        "median_R": float(np.median(R)),
    }


def overall_rows(rows: list[ReplayResult]) -> list[ReplayResult]:
    out = []
    for model in sorted({row.model for row in rows}):
        for season in ["ALL", "2024", "2025", "2026"]:
            parts = [
                row for row in rows
                if row.model == model and row.metric != "ALL" and row.season == season
            ]
            if not parts:
                continue
            out.append(weighted_result(model, "ALL", season, parts))
    return out


def weighted_result(model, metric, season, parts):
    weights = np.asarray([max(part.n, 0) for part in parts], dtype=np.float64)
    weights = np.where(weights > 0.0, weights, 1.0)
    return ReplayResult(
        model=model,
        metric=metric,
        season=season,
        n=int(sum(part.n for part in parts)),
        weight_sum=float(np.sum(weights)),
        nll=weighted_mean([part.nll for part in parts], weights),
        rmse=weighted_mean([part.rmse for part in parts], weights),
        mae=weighted_mean([part.mae for part in parts], weights),
        coverage90=weighted_mean([part.coverage90 for part in parts], weights),
        nis_mean=weighted_mean([part.nis_mean for part in parts], weights),
        std_innov_rms=weighted_mean([part.std_innov_rms for part in parts], weights),
        std_innov_mean=weighted_mean([part.std_innov_mean for part in parts], weights),
        lag1_std_innov_corr=weighted_mean(
            [part.lag1_std_innov_corr for part in parts], weights
        ),
        mean_R=weighted_mean([part.mean_R for part in parts], weights),
        median_R=weighted_mean([part.median_R for part in parts], weights),
        q=weighted_mean([part.q for part in parts], weights),
        r_scale=None,
    )


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def lag1_corr(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    x = values[:-1]
    y = values[1:]
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(rows[0]).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows, path, args, details, elapsed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in rows if row.metric == "ALL"]
    season_rows = [row for row in all_rows if row.season != "ALL"]
    wins = []
    for season in sorted({row.season for row in season_rows}):
        c = next(row for row in season_rows if row.season == season and row.model == "chimera_variance")
        h = next(row for row in season_rows if row.season == season and row.model == "incumbent_weight_heuristic")
        wins.append({
            "season": season,
            "nll_win": c.nll < h.nll,
            "rmse_win": c.rmse < h.rmse,
            "nis_closer": abs(c.nis_mean - 1.0) < abs(h.nis_mean - 1.0),
        })
    lines = [
        "# WNBA Kalman Replay",
        "",
        "Scalar per-metric random-walk Kalman replay on the WNBA DARKO "
        "game-level metric observation artifact. The Chimera lane injects "
        "`predict_variance()` as row-level `R_t`; the incumbent lane uses "
        "`sigma2 / sample_weight` with validation-tuned `sigma2` scale.",
        "",
        f"- Data: `{args.data}`",
        f"- Train through season: {args.train_through_season}",
        f"- Validation seasons: {args.val_start_season}-{args.val_through_season}",
        f"- Test seasons: {', '.join(sorted({row.season for row in season_rows}))}",
        f"- Chimera best iteration: {details['best_iteration']}",
        f"- Per-metric calibration groups: {details['group_count']}",
        f"- Runtime seconds: {elapsed:.2f}",
        "",
        "## Overall",
        "",
        "| model | season | n | NLL | RMSE | 90% cov | NIS mean | z RMS | lag1 z corr | mean R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row.model} | {row.season} | {row.n} | {row.nll:.3f} | "
            f"{row.rmse:.3f} | {row.coverage90:.3f} | {row.nis_mean:.3f} | "
            f"{row.std_innov_rms:.3f} | {row.lag1_std_innov_corr:.3f} | "
            f"{row.mean_R:.3f} |"
        )
    lines.extend([
        "",
        "## Season Wins",
        "",
        "| season | Chimera lower NLL | Chimera lower RMSE | Chimera NIS closer to 1 |",
        "|---:|---:|---:|---:|",
    ])
    for row in wins:
        lines.append(
            f"| {row['season']} | {row['nll_win']} | {row['rmse_win']} | "
            f"{row['nis_closer']} |"
        )
    nll_wins = sum(1 for row in wins if row["nll_win"])
    rmse_wins = sum(1 for row in wins if row["rmse_win"])
    nis_wins = sum(1 for row in wins if row["nis_closer"])
    verdict = (
        "does not clear the production replacement gate: Chimera variance "
        f"wins NLL in {nll_wins}/3 seasons, RMSE in {rmse_wins}/3 seasons, "
        f"and NIS closeness in {nis_wins}/3 seasons. Treat the variance as "
        "calibration-useful, but keep the incumbent R fallback for production."
    )
    if nll_wins >= 2 and rmse_wins >= 2 and nis_wins >= 2:
        verdict = (
            "clears the shadow replay gate on this artifact: Chimera variance "
            f"wins NLL in {nll_wins}/3 seasons, RMSE in {rmse_wins}/3 seasons, "
            f"and NIS closeness in {nis_wins}/3 seasons."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Verdict: {verdict}",
        "- This is still a game-metric observation replay, not a mutation of the "
        "production player DARKO filter. A production rollout should wire the "
        "same row-level `R_t` contract into that pipeline and retain an "
        "automatic incumbent fallback.",
    ])
    lines.extend([
        "",
        "## Metric Details",
        "",
        "| model | metric | season | n | NLL | RMSE | 90% cov | NIS mean | mean R | q | r scale |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row.metric == "ALL" or row.season != "ALL":
            continue
        r_scale = "" if row.r_scale is None else f"{row.r_scale:.3f}"
        lines.append(
            f"| {row.model} | {row.metric} | {row.season} | {row.n} | "
            f"{row.nll:.3f} | {row.rmse:.3f} | {row.coverage90:.3f} | "
            f"{row.nis_mean:.3f} | {row.mean_R:.3f} | {row.q:.6g} | "
            f"{r_scale} |"
        )
    lines.extend([
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(
            {
                "args": vars(args) | {
                    "data": str(args.data),
                    "output_csv": str(args.output_csv),
                    "output_summary": str(args.output_summary),
                },
                "details": details,
                "season_wins": wins,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
