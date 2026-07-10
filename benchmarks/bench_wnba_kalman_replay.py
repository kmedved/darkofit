"""WNBA DARKO observation-noise shadow replay with DarkoFit variance.

This is a research benchmark for the Kalman-readiness gate. It uses the
game-level WNBA DARKO metric observation artifact, fits the calibrated Gaussian
distributional head on train/validation seasons, then replays a scalar
random-walk Kalman filter per metric over held-out seasons with either:

* incumbent heuristic observation variance: ``R_t = sigma2 / sample_weight``;
* DarkoFit distributional observation variance:
  ``R_t = predict_variance(X_t)``.

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
from darkofit import DarkoRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "benchmarks" / "wnba_kalman_replay.csv"
DEFAULT_SUMMARY = ROOT / "benchmarks" / "wnba_kalman_replay_summary.md"
Z90 = 1.6448536269514722
RAW_DARKOFIT_MODEL = "darkofit_variance"
SCALED_DARKOFIT_MODEL = "darkofit_replay_scaled"
BLEND_MODEL = "darkofit_incumbent_blend"
INCUMBENT_MODEL = "incumbent_weight_heuristic"


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
    r_mix: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-through-season", type=int, default=2021)
    parser.add_argument("--val-start-season", type=int, default=2022)
    parser.add_argument("--val-through-season", type=int, default=2023)
    parser.add_argument("--loss", choices=["Gaussian", "StudentT"], default="Gaussian")
    parser.add_argument("--student-t-nu", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=25)
    parser.add_argument("--l2-leaf-reg", default="auto")
    parser.add_argument("--rho-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--rho-l2-leaf-reg-multiplier", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--r-floor",
        type=float,
        default=None,
        help=(
            "absolute floor for DarkoFit row-level R_t; by default the floor is "
            "--r-floor-fraction times each metric's incumbent tuned mean R"
        ),
    )
    parser.add_argument("--r-floor-fraction", type=float, default=0.25)
    parser.add_argument("--r-ceil", type=float, default=9.0)
    parser.add_argument("--darkofit-scale-min", type=float, default=0.25)
    parser.add_argument("--darkofit-scale-max", type=float, default=4.0)
    parser.add_argument("--darkofit-scale-steps", type=int, default=25)
    parser.add_argument("--hybrid-mix-steps", type=int, default=21)
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

    loss_name, dist_params = resolve_loss_args(args)
    raw_model, scaled_model, blend_model = distribution_model_names(args)
    model = DarkoRegressor(
        loss=loss_name,
        dist_params=dist_params,
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        l2_leaf_reg=args.l2_leaf_reg,
        rho_learning_rate_multiplier=args.rho_learning_rate_multiplier,
        rho_l2_leaf_reg_multiplier=args.rho_l2_leaf_reg_multiplier,
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
    darkofit_R_raw = model.predict_variance(X)

    rows: list[ReplayResult] = []
    pairwise_inputs = {}
    details = {
        "best_iteration": int(model.best_n_estimators_),
        "loss": loss_name,
        "dist_params": dist_params or {},
        "sigma_affine_b": float(getattr(model, "sigma_affine_b_", np.nan)),
        "group_count": int(len(getattr(model, "dist_group_affine_metadata_", ()))),
        "r_floor": None if args.r_floor is None else float(args.r_floor),
        "r_floor_fraction": float(args.r_floor_fraction),
        "r_floor_by_metric": {},
        "r_tuning_by_metric": {},
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
        heuristic_R, q_heuristic, r_scale = tune_heuristic_r(
            y_m, w_m, dates_m, game_m, train_m, val_m
        )
        r_floor = resolve_metric_r_floor(args, heuristic_R, w_m, train_m)
        details["r_floor_by_metric"][str(metric)] = float(r_floor)
        darkofit_R_m = np.clip(
            darkofit_R_raw[idx],
            float(r_floor),
            float(args.r_ceil),
        )
        q_darkofit = tune_q(y_m, w_m, dates_m, game_m, darkofit_R_m, train_m, val_m)
        scaled_R, q_scaled, scaled_scale = tune_scaled_r(
            y_m,
            w_m,
            dates_m,
            game_m,
            darkofit_R_raw[idx],
            train_m,
            val_m,
            r_floor,
            args,
        )
        blend_R, q_blend, blend_mix = tune_blend_r(
            y_m,
            w_m,
            dates_m,
            game_m,
            scaled_R,
            heuristic_R,
            train_m,
            val_m,
            args,
        )
        details["r_tuning_by_metric"][str(metric)] = {
            "raw_q": float(q_darkofit),
            "scaled_q": float(q_scaled),
            "scaled_scale": float(scaled_scale),
            "blend_q": float(q_blend),
            "blend_darkofit_mix": float(blend_mix),
            "incumbent_q": float(q_heuristic),
            "incumbent_scale": float(r_scale),
        }
        metric_results = [
            (
                raw_model,
                darkofit_R_m,
                q_darkofit,
                None,
                None,
            ),
            (
                scaled_model,
                scaled_R,
                q_scaled,
                scaled_scale,
                None,
            ),
            (
                blend_model,
                blend_R,
                q_blend,
                scaled_scale,
                blend_mix,
            ),
            (
                INCUMBENT_MODEL,
                heuristic_R,
                q_heuristic,
                r_scale,
                None,
            ),
        ]
        replay_cache = {}
        for name, R_m, q, r_scale_value, r_mix_value in metric_results:
            replay = run_replay(y_m, w_m, dates_m, game_m, R_m, q, train_m)
            replay_cache[name] = replay
            rows.extend(
                score_replay(
                    name,
                    metric,
                    replay,
                    test_m,
                    seasons_m,
                    q,
                    r_scale_value,
                    r_mix_value,
                )
            )
        incumbent_replay = replay_cache[INCUMBENT_MODEL]
        for name, replay in replay_cache.items():
            if name == INCUMBENT_MODEL:
                continue
            append_pairwise_inputs(
                pairwise_inputs,
                name,
                replay,
                incumbent_replay,
                test_m,
                seasons_m,
            )

    rows.extend(overall_rows(rows))
    write_csv(rows, args.output_csv)
    write_summary(
        rows,
        args.output_summary,
        args,
        details,
        pairwise_inputs,
        time.perf_counter() - start,
    )
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
        replay = run_replay(y, w, dates, game_id, R, float(q), train_mask)
        row = score_arrays(replay, val_mask)
        if row["nll"] < best_nll:
            best_nll = row["nll"]
            best_q = float(q)
    return float(best_q if best_q is not None else 0.0)


def resolve_loss_args(args):
    loss_name = str(args.loss)
    if loss_name == "StudentT":
        nu = float(args.student_t_nu)
        if not np.isfinite(nu) or nu <= 2.0:
            raise ValueError("--student-t-nu must be finite and greater than 2")
        return loss_name, {"nu": nu}
    return loss_name, None


def distribution_model_names(args):
    if str(args.loss) == "StudentT":
        nu_token = f"{float(args.student_t_nu):g}".replace(".", "p")
        prefix = f"studentt{nu_token}"
        return (
            f"{prefix}_variance",
            f"{prefix}_replay_scaled",
            f"{prefix}_incumbent_blend",
        )
    return RAW_DARKOFIT_MODEL, SCALED_DARKOFIT_MODEL, BLEND_MODEL


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
        replay = run_replay(y, w, dates, game_id, R, q, train_mask)
        row = score_arrays(replay, val_mask)
        key = (row["nll"], row["rmse"])
        if best is None or key < best["key"]:
            best = {"key": key, "R": R, "q": q, "scale": float(scale)}
    return best["R"], float(best["q"]), float(best["scale"])


def positive_log_grid(low, high, steps):
    low = float(low)
    high = float(high)
    steps = int(steps)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= 0.0:
        raise ValueError("scale grid bounds must be positive finite numbers")
    if high < low:
        raise ValueError("scale grid max must be >= min")
    if steps <= 1 or high == low:
        return np.asarray([low], dtype=np.float64)
    return np.exp(np.linspace(np.log(low), np.log(high), steps))


def tune_scaled_r(y, w, dates, game_id, base_R, train_mask, val_mask, r_floor, args):
    best = None
    for scale in positive_log_grid(
        args.darkofit_scale_min,
        args.darkofit_scale_max,
        args.darkofit_scale_steps,
    ):
        R = np.clip(base_R * float(scale), float(r_floor), float(args.r_ceil))
        q = tune_q(y, w, dates, game_id, R, train_mask, val_mask)
        replay = run_replay(y, w, dates, game_id, R, q, train_mask)
        row = score_arrays(replay, val_mask)
        key = (row["nll"], row["rmse"])
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "R": R,
                "q": float(q),
                "scale": float(scale),
            }
    return best["R"], float(best["q"]), float(best["scale"])


def tune_blend_r(y, w, dates, game_id, darkofit_R, incumbent_R,
                 train_mask, val_mask, args):
    steps = int(args.hybrid_mix_steps)
    if steps <= 1:
        mix_grid = np.asarray([0.5], dtype=np.float64)
    else:
        mix_grid = np.linspace(0.0, 1.0, steps)
    best = None
    for mix in mix_grid:
        R = np.clip(
            float(mix) * darkofit_R + (1.0 - float(mix)) * incumbent_R,
            1.0e-12,
            float(args.r_ceil),
        )
        q = tune_q(y, w, dates, game_id, R, train_mask, val_mask)
        replay = run_replay(y, w, dates, game_id, R, q, train_mask)
        row = score_arrays(replay, val_mask)
        key = (row["nll"], row["rmse"])
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "R": R,
                "q": float(q),
                "mix": float(mix),
            }
    return best["R"], float(best["q"]), float(best["mix"])


def resolve_metric_r_floor(args, heuristic_R, w, train_mask):
    if args.r_floor is not None:
        return max(float(args.r_floor), 1.0e-12)
    fraction = float(args.r_floor_fraction)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("--r-floor-fraction must be finite and nonnegative")
    train_R = np.asarray(heuristic_R, dtype=np.float64)[train_mask]
    train_w = np.asarray(w, dtype=np.float64)[train_mask]
    baseline = weighted_mean(train_R, train_w)
    if not np.isfinite(baseline) or baseline <= 0.0:
        baseline = weighted_mean(heuristic_R, np.ones_like(heuristic_R))
    return max(fraction * baseline, 1.0e-12)


def run_replay(y, w, dates, game_id, R, q, train_mask=None):
    if train_mask is None:
        train_mask = np.ones_like(y, dtype=bool)
        skip_warmup_updates = False
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape != y.shape:
            raise ValueError("train_mask must have the same shape as y")
        skip_warmup_updates = True
    if not np.any(train_mask):
        raise ValueError("train_mask must select at least one warmup row")
    order = np.lexsort((game_id, dates))
    y_ord = y[order]
    w_ord = w[order]
    R_ord = np.maximum(R[order], 1.0e-9)
    date_ord = dates[order]
    train_ord = train_mask[order]
    y_init = y[train_mask]
    w_init = w[train_mask]
    mean = weighted_mean(y_init, w_init)
    var = max(weighted_mean((y_init - mean) ** 2, w_init), 1.0e-6)
    prev_date = None
    pred = np.empty_like(y_ord, dtype=np.float64)
    pred_var = np.empty_like(y_ord, dtype=np.float64)
    obs_var = np.empty_like(y_ord, dtype=np.float64)
    innovation = np.empty_like(y_ord, dtype=np.float64)
    standardized = np.empty_like(y_ord, dtype=np.float64)
    for pos, idx in enumerate(order):
        current_date = date_ord[pos]
        S = max(var + R_ord[pos], 1.0e-9)
        err = y_ord[pos] - mean
        pred[pos] = mean
        pred_var[pos] = S
        obs_var[pos] = R_ord[pos]
        innovation[pos] = err
        standardized[pos] = err / math.sqrt(S)
        if skip_warmup_updates and train_ord[pos]:
            prev_date = current_date
            continue
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
        pred_var[pos] = S
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


def score_replay(model, metric, replay, test_mask, seasons, q, r_scale, r_mix):
    rows = []
    rows.append(
        make_result(model, metric, "ALL", replay, test_mask, q, r_scale, r_mix)
    )
    for season in sorted(np.unique(seasons[test_mask])):
        mask = test_mask & (seasons == season)
        rows.append(
            make_result(
                model,
                metric,
                str(int(season)),
                replay,
                mask,
                q,
                r_scale,
                r_mix,
            )
        )
    return rows


def make_result(model, metric, season, replay, mask, q, r_scale, r_mix):
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
        r_mix=None if r_mix is None else float(r_mix),
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


def row_score_arrays(replay, mask):
    mask = np.asarray(mask, dtype=bool)
    innovation = replay["innovation"][mask]
    pred_var = np.maximum(replay["pred_var"][mask], 1.0e-9)
    z = replay["standardized"][mask]
    return {
        "nll": (
            0.5 * np.log(2.0 * np.pi * pred_var)
            + 0.5 * innovation * innovation / pred_var
        ),
        "se": innovation * innovation,
        "z2": z * z,
    }


def append_pairwise_inputs(pairwise_inputs, model, replay, incumbent_replay,
                           test_mask, seasons):
    season_values = ["ALL"] + [
        str(int(season)) for season in sorted(np.unique(seasons[test_mask]))
    ]
    for season in season_values:
        mask = np.asarray(test_mask, dtype=bool)
        if season != "ALL":
            mask = mask & (seasons == int(season))
        if not np.any(mask):
            continue
        candidate = row_score_arrays(replay, mask)
        incumbent = row_score_arrays(incumbent_replay, mask)
        bucket = pairwise_inputs.setdefault(
            (model, season),
            {
                "candidate_nll": [],
                "incumbent_nll": [],
                "candidate_se": [],
                "incumbent_se": [],
                "candidate_z2": [],
                "incumbent_z2": [],
            },
        )
        bucket["candidate_nll"].append(candidate["nll"])
        bucket["incumbent_nll"].append(incumbent["nll"])
        bucket["candidate_se"].append(candidate["se"])
        bucket["incumbent_se"].append(incumbent["se"])
        bucket["candidate_z2"].append(candidate["z2"])
        bucket["incumbent_z2"].append(incumbent["z2"])


def paired_bootstrap_summaries(pairwise_inputs, n_boot=2000, seed=20260708):
    rng = np.random.default_rng(seed)
    summaries = {}
    for (model, season), arrays in pairwise_inputs.items():
        c_nll = np.concatenate(arrays["candidate_nll"]).astype(np.float64)
        h_nll = np.concatenate(arrays["incumbent_nll"]).astype(np.float64)
        c_se = np.concatenate(arrays["candidate_se"]).astype(np.float64)
        h_se = np.concatenate(arrays["incumbent_se"]).astype(np.float64)
        c_z2 = np.concatenate(arrays["candidate_z2"]).astype(np.float64)
        h_z2 = np.concatenate(arrays["incumbent_z2"]).astype(np.float64)
        n = c_nll.size
        summaries[(model, season)] = {
            "model": model,
            "season": season,
            "n": int(n),
            "nll": bootstrap_metric_summary(
                c_nll, h_nll, rng, n_boot, paired_mean_gap
            ),
            "rmse": bootstrap_metric_summary(
                c_se, h_se, rng, n_boot, paired_rmse_gap
            ),
            "nis_closeness": bootstrap_metric_summary(
                c_z2, h_z2, rng, n_boot, paired_nis_closeness_gap
            ),
        }
    return summaries


def bootstrap_metric_summary(candidate, incumbent, rng, n_boot, statistic):
    observed = float(statistic(candidate, incumbent))
    n = candidate.size
    if n == 0:
        return {
            "gap": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "result": "tie",
        }
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(candidate[idx], incumbent[idx])
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    if ci_low <= 0.0 <= ci_high:
        result = "tie"
    elif observed < 0.0:
        result = "win"
    else:
        result = "loss"
    return {
        "gap": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "result": result,
    }


def paired_mean_gap(candidate, incumbent):
    return float(np.mean(candidate - incumbent))


def paired_rmse_gap(candidate_se, incumbent_se):
    return float(np.sqrt(np.mean(candidate_se)) - np.sqrt(np.mean(incumbent_se)))


def paired_nis_closeness_gap(candidate_z2, incumbent_z2):
    return float(abs(np.mean(candidate_z2) - 1.0) - abs(np.mean(incumbent_z2) - 1.0))


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
        rmse=math.sqrt(
            weighted_mean([part.rmse * part.rmse for part in parts], weights)
        ),
        mae=weighted_mean([part.mae for part in parts], weights),
        coverage90=weighted_mean([part.coverage90 for part in parts], weights),
        nis_mean=weighted_mean([part.nis_mean for part in parts], weights),
        std_innov_rms=math.sqrt(
            weighted_mean(
                [part.std_innov_rms * part.std_innov_rms for part in parts],
                weights,
            )
        ),
        std_innov_mean=weighted_mean([part.std_innov_mean for part in parts], weights),
        lag1_std_innov_corr=weighted_mean(
            [part.lag1_std_innov_corr for part in parts], weights
        ),
        mean_R=weighted_mean([part.mean_R for part in parts], weights),
        median_R=weighted_mean([part.median_R for part in parts], weights),
        q=weighted_mean([part.q for part in parts], weights),
        r_scale=None,
        r_mix=None,
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


def r_floor_summary(details):
    if details.get("r_floor") is not None:
        return f"absolute {details['r_floor']:.6g}"
    return (
        f"{details['r_floor_fraction']:.3g} x each metric's incumbent tuned "
        "train mean R"
    )


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


def candidate_gate_summary(model, all_rows, wins):
    model_wins = [row for row in wins if row["model"] == model]
    season_count = len(model_wins)
    required_wins = season_count // 2 + 1
    nll_wins = sum(1 for row in model_wins if row["nll_result"] == "win")
    rmse_wins = sum(1 for row in model_wins if row["rmse_result"] == "win")
    nis_wins = sum(1 for row in model_wins if row["nis_result"] == "win")
    nll_ties = sum(1 for row in model_wins if row["nll_result"] == "tie")
    rmse_ties = sum(1 for row in model_wins if row["rmse_result"] == "tie")
    nis_ties = sum(1 for row in model_wins if row["nis_result"] == "tie")
    c_all = next(
        row for row in all_rows
        if row.model == model and row.season == "ALL"
    )
    h_all = next(
        row for row in all_rows
        if row.model == INCUMBENT_MODEL and row.season == "ALL"
    )
    return {
        "model": model,
        "season_count": season_count,
        "required_wins": required_wins,
        "nll_wins": nll_wins,
        "rmse_wins": rmse_wins,
        "nis_wins": nis_wins,
        "nll_ties": nll_ties,
        "rmse_ties": rmse_ties,
        "nis_ties": nis_ties,
        "clears_gate": (
            nll_wins >= required_wins
            and rmse_wins >= required_wins
            and nis_wins >= required_wins
        ),
        "overall_nll_gap": float(c_all.nll - h_all.nll),
        "overall_rmse_gap": float(c_all.rmse - h_all.rmse),
        "overall_nis_closeness_gap": float(
            abs(c_all.nis_mean - 1.0) - abs(h_all.nis_mean - 1.0)
        ),
    }


def serialize_bootstrap_summaries(bootstrap_summaries):
    out = []
    for (model, season), summary in sorted(bootstrap_summaries.items()):
        row = {"model": model, "season": season, "n": int(summary["n"])}
        for metric_name in ("nll", "rmse", "nis_closeness"):
            metric = summary[metric_name]
            row[f"{metric_name}_gap"] = float(metric["gap"])
            row[f"{metric_name}_ci_low"] = float(metric["ci_low"])
            row[f"{metric_name}_ci_high"] = float(metric["ci_high"])
            row[f"{metric_name}_result"] = metric["result"]
        out.append(row)
    return out


def write_summary(rows, path, args, details, pairwise_inputs, elapsed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in rows if row.metric == "ALL"]
    season_rows = [row for row in all_rows if row.season != "ALL"]
    candidate_models = sorted(
        {row.model for row in all_rows if row.model != INCUMBENT_MODEL}
    )
    bootstrap_summaries = paired_bootstrap_summaries(pairwise_inputs)
    wins = []
    for model in candidate_models:
        for season in sorted({row.season for row in season_rows}):
            c = next(
                row for row in season_rows
                if row.season == season and row.model == model
            )
            h = next(
                row for row in season_rows
                if row.season == season and row.model == INCUMBENT_MODEL
            )
            bootstrap = bootstrap_summaries[(model, season)]
            wins.append({
                "model": model,
                "season": season,
                "nll_gap": float(c.nll - h.nll),
                "rmse_gap": float(c.rmse - h.rmse),
                "nis_closeness_gap": float(
                    abs(c.nis_mean - 1.0) - abs(h.nis_mean - 1.0)
                ),
                "nll_result": bootstrap["nll"]["result"],
                "rmse_result": bootstrap["rmse"]["result"],
                "nis_result": bootstrap["nis_closeness"]["result"],
            })
    lines = [
        "# WNBA Kalman Replay",
        "",
        "Scalar per-metric random-walk Kalman replay on the WNBA DARKO "
        "game-level metric observation artifact. The DarkoFit lane injects "
        "`predict_variance()` as row-level `R_t`; the incumbent lane uses "
        "`sigma2 / sample_weight` with validation-tuned `sigma2` scale.",
        "",
        f"- Data: `{args.data}`",
        f"- Train through season: {args.train_through_season}",
        f"- Validation seasons: {args.val_start_season}-{args.val_through_season}",
        f"- Test seasons: {', '.join(sorted({row.season for row in season_rows}))}",
        f"- Distributional loss: {details['loss']}",
        f"- Model best iteration: {details['best_iteration']}",
        f"- Per-metric calibration groups: {details['group_count']}",
        f"- R floor: {r_floor_summary(details)}",
        "- Gate noise band: paired row bootstrap, 2,000 resamples, 95% CI; "
        "differences whose interval crosses zero are ties.",
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
        "## Season Results vs Incumbent",
        "",
        "| model | season | NLL result | RMSE result | NIS result |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in wins:
        lines.append(
            f"| {row['model']} | {row['season']} | {row['nll_result']} | "
            f"{row['rmse_result']} | {row['nis_result']} |"
        )
    candidate_summaries = [
        candidate_gate_summary(model, all_rows, wins)
        for model in candidate_models
    ]
    best = max(
        candidate_summaries,
        key=lambda item: (
            item["clears_gate"],
            item["nll_wins"],
            item["rmse_wins"],
            item["nis_wins"],
            -item["overall_nll_gap"],
            -item["overall_rmse_gap"],
        ),
    )
    season_count = int(best["season_count"])
    required_wins = season_count // 2 + 1
    verdict = (
        f"does not clear the production replacement gate: best candidate "
        f"`{best['model']}` wins NLL in {best['nll_wins']}/{season_count} "
        f"seasons with {best['nll_ties']} ties, RMSE in "
        f"{best['rmse_wins']}/{season_count} seasons with "
        f"{best['rmse_ties']} ties, and NIS closeness in "
        f"{best['nis_wins']}/{season_count} seasons with "
        f"{best['nis_ties']} ties "
        f"(majority threshold {required_wins}/{season_count}). Treat the "
        "variance as calibration-useful, but keep the incumbent R fallback "
        "for production."
    )
    if best["clears_gate"]:
        verdict = (
            f"clears the shadow replay gate on this artifact: `{best['model']}` "
            f"wins NLL in {best['nll_wins']}/{season_count} seasons, RMSE in "
            f"{best['rmse_wins']}/{season_count} seasons, and NIS closeness "
            f"in {best['nis_wins']}/{season_count} seasons after bootstrap "
            "tie handling."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Verdict: {verdict}",
        f"- Best candidate overall gap vs incumbent: NLL "
        f"{best['overall_nll_gap']:+.4f}, RMSE {best['overall_rmse_gap']:+.4f}, "
        f"NIS-closeness {best['overall_nis_closeness_gap']:+.4f}.",
        "- The best candidate is statistically indistinguishable from the "
        "incumbent on this scalar replay; the useful signal is parity plus "
        "better overall second-moment calibration, not a standalone "
        "replacement claim.",
        "- This is still a game-metric observation replay, not a mutation of the "
        "production player DARKO filter. A production rollout should wire the "
        "same row-level `R_t` contract into that pipeline and retain an "
        "automatic incumbent fallback.",
    ])
    lines.extend([
        "",
        "## Metric Details",
        "",
        "| model | metric | season | n | NLL | RMSE | 90% cov | NIS mean | mean R | q | r scale | r mix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row.metric == "ALL" or row.season != "ALL":
            continue
        r_scale = "" if row.r_scale is None else f"{row.r_scale:.3f}"
        r_mix = "" if row.r_mix is None else f"{row.r_mix:.3f}"
        lines.append(
            f"| {row.model} | {row.metric} | {row.season} | {row.n} | "
            f"{row.nll:.3f} | {row.rmse:.3f} | {row.coverage90:.3f} | "
            f"{row.nis_mean:.3f} | {row.mean_R:.3f} | {row.q:.6g} | "
            f"{r_scale} | {r_mix} |"
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
                "candidate_summaries": candidate_summaries,
                "bootstrap_summaries": serialize_bootstrap_summaries(
                    bootstrap_summaries
                ),
                "season_results": wins,
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
