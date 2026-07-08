"""WNBA real-data distributional calibration check.

This benchmark uses WNBA DARKO game-level metric observations as a real
observation-noise validation set for ChimeraBoost's Gaussian head.  The target
column is named ``z_observed`` in the source artifact, but it is the transformed
metric observation scale rather than a globally standard-normal target.  The
benchmark is offline and read-only with respect to the WNBA repo.

Default data source:
    /Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/
        calculated_data/research/observation_covariance_measurement/
        game_metric_observations.parq

Example:
    PYTHONPATH=. /Users/kmedved/.venvs/darko311/bin/python \\
        benchmarks/bench_wnba_realdata_distributional.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from chimeraboost import ChimeraBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_DATA = Path(
    "/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/"
    "calculated_data/research/observation_covariance_measurement/"
    "game_metric_observations.parq"
)
DEFAULT_CSV = ROOT / "benchmarks" / "wnba_realdata_distributional.csv"
DEFAULT_MARKDOWN = (
    ROOT / "benchmarks" / "wnba_realdata_distributional_summary.md"
)
Z90 = 1.6448536269514722
SQRT_PI_INV = 1.0 / math.sqrt(math.pi)
RECENT_WINDOWS = (7, 30, 90, 365)


@dataclass
class Result:
    model: str
    status: str
    reason: str
    train_rows: int
    val_rows: int
    test_rows: int
    n_features: int
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    best_iteration: int | None = None
    sigma_scale: float | None = None
    sigma_affine_a: float | None = None
    sigma_affine_b: float | None = None
    rmse_mu: float | None = None
    nll: float | None = None
    crps: float | None = None
    coverage90: float | None = None
    width90: float | None = None
    std_resid_rms: float | None = None
    std_resid_mean: float | None = None
    sigma_mean: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    sigma_abs_resid_corr: float | None = None
    coverage90_by_sigma: str | None = None
    std_rms_by_sigma: str | None = None
    z2_by_sigma: str | None = None
    config: str | None = None


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.average(values, weights=weights))


def _weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return math.sqrt(_weighted_mean(values * values, weights))


def _weighted_corr(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    x_mean = _weighted_mean(x, weights)
    y_mean = _weighted_mean(y, weights)
    xc = x - x_mean
    yc = y - y_mean
    x_var = _weighted_mean(xc * xc, weights)
    y_var = _weighted_mean(yc * yc, weights)
    if x_var <= 0.0 or y_var <= 0.0:
        return float("nan")
    return float(_weighted_mean(xc * yc, weights) / math.sqrt(x_var * y_var))


def _parse_grid(value, *, cast=float):
    if value is None or str(value).strip() == "":
        return []
    out = []
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        out.append(token if token == "auto" else cast(token))
    return out


def _dedupe(values):
    out = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _with_arg_overrides(args, **overrides):
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _config_string(args):
    return (
        f"lr={args.learning_rate}, leaves={args.num_leaves}, "
        f"min_child={args.min_child_samples}, l2={args.l2_leaf_reg}, "
        f"rho_lr={args.rho_learning_rate_multiplier}, "
        f"rho_l2={args.rho_l2_leaf_reg_multiplier}"
    )


def _gate_stats(metric_sigma_slices):
    deviations = [
        abs(float(row["std_resid_rms"]) - 1.0)
        for row in metric_sigma_slices
        if row.get("std_resid_rms") is not None
        and np.isfinite(float(row["std_resid_rms"]))
    ]
    if not deviations:
        return {
            "max_metric_rms_dev": float("inf"),
            "mean_metric_rms_dev": float("inf"),
            "failed_metric_bins": 0,
        }
    return {
        "max_metric_rms_dev": float(np.max(deviations)),
        "mean_metric_rms_dev": float(np.mean(deviations)),
        "failed_metric_bins": int(np.sum(np.asarray(deviations) > 0.05)),
    }


def _prior_feature_columns():
    cols = [
        "all_prior_weight",
        "all_prior_mean",
        "all_prior_std",
        "season_prior_weight",
        "season_prior_mean",
        "season_prior_std",
    ]
    for window in RECENT_WINDOWS:
        prefix = f"recent{window}"
        cols.extend([
            f"{prefix}_w",
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_std_to_all",
            f"{prefix}_std_to_season",
        ])
    return cols


def _normal_nll(y, mu, sigma, weights):
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    z = (np.asarray(y, dtype=np.float64) - mu) / sigma
    row = 0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z * z
    return _weighted_mean(row, weights)


def _normal_crps(y, mu, sigma, weights):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    row = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - SQRT_PI_INV)
    return _weighted_mean(row, weights)


def _bin_summary(y, mu, sigma, weights, n_bins):
    order = np.argsort(sigma)
    y = np.asarray(y, dtype=np.float64)[order]
    mu = np.asarray(mu, dtype=np.float64)[order]
    sigma = np.clip(np.asarray(sigma, dtype=np.float64)[order], 1e-12, None)
    weights = np.asarray(weights, dtype=np.float64)[order]
    covered = (y >= mu - Z90 * sigma) & (y <= mu + Z90 * sigma)
    std_resid = (y - mu) / sigma
    cov_parts = []
    rms_parts = []
    z2_parts = []
    for idx in np.array_split(np.arange(y.shape[0]), int(n_bins)):
        if idx.size == 0:
            continue
        cov_parts.append(_weighted_mean(covered[idx].astype(float), weights[idx]))
        rms_parts.append(_weighted_rms(std_resid[idx], weights[idx]))
        z2_parts.append(_weighted_mean(std_resid[idx] ** 2, weights[idx]))
    return (
        "/".join(f"{value:.3f}" for value in cov_parts),
        "/".join(f"{value:.3f}" for value in rms_parts),
        "/".join(f"{value:.3f}" for value in z2_parts),
    )


def _normal_cdf(z):
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))


def _prediction_frame(df_test, y, mu, sigma, weights):
    frame = df_test[["date", "season", "game_id", "metric"]].copy()
    frame["y"] = np.asarray(y, dtype=np.float64)
    frame["mu"] = np.asarray(mu, dtype=np.float64)
    frame["sigma"] = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    frame["sample_weight"] = np.asarray(weights, dtype=np.float64)
    frame["std_resid"] = (frame["y"] - frame["mu"]) / frame["sigma"]
    frame["covered90"] = (
        (frame["y"] >= frame["mu"] - Z90 * frame["sigma"])
        & (frame["y"] <= frame["mu"] + Z90 * frame["sigma"])
    )
    frame["pit"] = _normal_cdf(frame["std_resid"].to_numpy(dtype=np.float64))
    return frame


def _weighted_histogram(values, weights, bins):
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0), weights=weights)
    total = float(np.sum(counts))
    if total <= 0.0:
        return "nan", float("nan")
    fractions = counts / total
    return "/".join(f"{value:.3f}" for value in fractions), edges


def _weighted_ks_uniform(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0.0 or values.size == 0:
        return float("nan")
    cdf = np.cumsum(weights) / total
    prev = np.concatenate(([0.0], cdf[:-1]))
    return float(np.max(np.maximum(np.abs(cdf - values), np.abs(values - prev))))


def _metric_sigma_slice_rows(frame, n_bins=3):
    rows = []
    for metric, sub in frame.groupby("metric", observed=True):
        order = np.argsort(sub["sigma"].to_numpy(dtype=np.float64))
        ordered = sub.iloc[order]
        for bin_idx, idx in enumerate(np.array_split(np.arange(len(ordered)), n_bins), 1):
            if idx.size == 0:
                continue
            part = ordered.iloc[idx]
            weights = part["sample_weight"].to_numpy(dtype=np.float64)
            z = part["std_resid"].to_numpy(dtype=np.float64)
            rows.append({
                "metric": str(metric),
                "sigma_bin": int(bin_idx),
                "n": int(part.shape[0]),
                "coverage90": _weighted_mean(
                    part["covered90"].astype(float).to_numpy(dtype=np.float64),
                    weights,
                ),
                "std_resid_rms": _weighted_rms(z, weights),
                "z2": _weighted_mean(z * z, weights),
                "sigma_mean": _weighted_mean(
                    part["sigma"].to_numpy(dtype=np.float64), weights
                ),
            })
    return rows


def _lag1_rows(frame):
    rows = []
    pairs = []
    for metric, sub in frame.sort_values(
        ["metric", "date", "game_id"]
    ).groupby("metric", observed=True):
        z = sub["std_resid"].to_numpy(dtype=np.float64)
        weights = sub["sample_weight"].to_numpy(dtype=np.float64)
        if z.size < 2:
            corr = float("nan")
            n_pairs = 0
        else:
            corr = _weighted_corr(z[:-1], z[1:], weights[1:])
            n_pairs = z.size - 1
            pairs.append((z[:-1], z[1:], weights[1:]))
        rows.append({"metric": str(metric), "n_pairs": int(n_pairs), "lag1": corr})
    if pairs:
        prev = np.concatenate([item[0] for item in pairs])
        curr = np.concatenate([item[1] for item in pairs])
        weights = np.concatenate([item[2] for item in pairs])
        rows.insert(0, {
            "metric": "pooled_metric_order",
            "n_pairs": int(prev.size),
            "lag1": _weighted_corr(prev, curr, weights),
        })
    return rows


def _diagnostics_from_frame(frame, sigma_bins):
    pit_hist, _ = _weighted_histogram(
        frame["pit"].to_numpy(dtype=np.float64),
        frame["sample_weight"].to_numpy(dtype=np.float64),
        bins=10,
    )
    return {
        "metric_sigma_slices": _metric_sigma_slice_rows(frame, n_bins=3),
        "lag1": _lag1_rows(frame),
        "pit_histogram": pit_hist,
        "pit_ks": _weighted_ks_uniform(
            frame["pit"].to_numpy(dtype=np.float64),
            frame["sample_weight"].to_numpy(dtype=np.float64),
        ),
        "sigma_bins": int(sigma_bins),
    }


def _score(name, train_rows, val_rows, X_test, y_test, w_test,
           fit_seconds, predict_seconds, mu, sigma, *, best_iteration=None,
           sigma_scale=None, sigma_affine_a=None, sigma_affine_b=None, bins=5,
           config=None):
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    mu = np.asarray(mu, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    w_test = np.asarray(w_test, dtype=np.float64)
    std_resid = (y_test - mu) / sigma
    covered = (y_test >= mu - Z90 * sigma) & (y_test <= mu + Z90 * sigma)
    cov_by_sigma, rms_by_sigma, z2_by_sigma = _bin_summary(
        y_test, mu, sigma, w_test, bins
    )
    return Result(
        model=name,
        status="ok",
        reason="",
        train_rows=int(train_rows),
        val_rows=int(val_rows),
        test_rows=int(X_test.shape[0]),
        n_features=int(X_test.shape[1]),
        fit_seconds=float(fit_seconds),
        predict_seconds=float(predict_seconds),
        best_iteration=(
            None if best_iteration is None else int(best_iteration)
        ),
        sigma_scale=None if sigma_scale is None else float(sigma_scale),
        sigma_affine_a=(
            None if sigma_affine_a is None else float(sigma_affine_a)
        ),
        sigma_affine_b=(
            None if sigma_affine_b is None else float(sigma_affine_b)
        ),
        rmse_mu=_weighted_rms(y_test - mu, w_test),
        nll=_normal_nll(y_test, mu, sigma, w_test),
        crps=_normal_crps(y_test, mu, sigma, w_test),
        coverage90=_weighted_mean(covered.astype(float), w_test),
        width90=_weighted_mean(2.0 * Z90 * sigma, w_test),
        std_resid_rms=_weighted_rms(std_resid, w_test),
        std_resid_mean=_weighted_mean(std_resid, w_test),
        sigma_mean=_weighted_mean(sigma, w_test),
        sigma_min=float(np.min(sigma)),
        sigma_max=float(np.max(sigma)),
        sigma_abs_resid_corr=_weighted_corr(
            sigma, np.abs(y_test - mu), w_test
        ),
        coverage90_by_sigma=cov_by_sigma,
        std_rms_by_sigma=rms_by_sigma,
        z2_by_sigma=z2_by_sigma,
        config=config,
    )


def _weighted_daily(values):
    weights = values["sample_weight"].to_numpy(dtype=np.float64)
    z = values["z_observed"].to_numpy(dtype=np.float64)
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        mean = float(np.mean(z))
        z2 = float(np.mean(z * z))
    else:
        mean = float(np.average(z, weights=weights))
        z2 = float(np.average(z * z, weights=weights))
    return pd.Series({"day_w": w_sum, "day_z": mean, "day_z2": z2})


def _add_prior_features(df):
    daily = (
        df.groupby(["metric", "date"], observed=True)
        .apply(_weighted_daily, include_groups=False)
        .reset_index()
        .sort_values(["metric", "date"])
    )
    daily["season"] = daily["date"].dt.year
    daily["day_wz"] = daily["day_w"] * daily["day_z"]
    daily["day_wz2"] = daily["day_w"] * daily["day_z2"]

    for prefix, group_cols in [
        ("all", ["metric"]),
        ("season", ["metric", "season"]),
    ]:
        group = daily.groupby(group_cols, observed=True)
        prior_w = group["day_w"].cumsum() - daily["day_w"]
        prior_wz = group["day_wz"].cumsum() - daily["day_wz"]
        prior_wz2 = group["day_wz2"].cumsum() - daily["day_wz2"]
        mean = np.divide(
            prior_wz,
            prior_w,
            out=np.zeros_like(prior_wz, dtype=np.float64),
            where=prior_w > 0.0,
        )
        second = np.divide(
            prior_wz2,
            prior_w,
            out=np.ones_like(prior_wz2, dtype=np.float64),
            where=prior_w > 0.0,
        )
        var = np.maximum(second - mean * mean, 1e-6)
        daily[f"{prefix}_prior_weight"] = prior_w
        daily[f"{prefix}_prior_mean"] = mean
        daily[f"{prefix}_prior_std"] = np.sqrt(var)

    for window in RECENT_WINDOWS:
        daily[f"recent{window}_w"] = 0.0
        daily[f"recent{window}_mean"] = 0.0
        daily[f"recent{window}_std"] = 1.0
    for _, idx in daily.groupby("metric", observed=True).groups.items():
        sub = daily.loc[idx].sort_values("date")
        for window in RECENT_WINDOWS:
            prefix = f"recent{window}"
            roll_w = (
                sub["day_w"].shift(1).rolling(window, min_periods=1).sum()
            )
            roll_wz = (
                sub["day_wz"].shift(1).rolling(window, min_periods=1).sum()
            )
            roll_wz2 = (
                sub["day_wz2"].shift(1).rolling(window, min_periods=1).sum()
            )
            mean = np.divide(
                roll_wz,
                roll_w,
                out=np.zeros_like(roll_wz, dtype=np.float64),
                where=roll_w > 0.0,
            )
            second = np.divide(
                roll_wz2,
                roll_w,
                out=np.ones_like(roll_wz2, dtype=np.float64),
                where=roll_w > 0.0,
            )
            var = np.maximum(second - mean * mean, 1e-6)
            daily.loc[sub.index, f"{prefix}_w"] = roll_w.fillna(0.0)
            daily.loc[sub.index, f"{prefix}_mean"] = pd.Series(mean).fillna(0.0)
            daily.loc[sub.index, f"{prefix}_std"] = np.sqrt(
                pd.Series(var).fillna(1.0)
            )

    rolling_cols = [
        col for col in _prior_feature_columns()
        if not col.endswith("_std_to_all") and not col.endswith("_std_to_season")
    ]
    merged = df.merge(
        daily[["metric", "date", *rolling_cols]],
        on=["metric", "date"],
        how="left",
        validate="many_to_one",
    )
    for col in rolling_cols:
        if col.endswith("_std"):
            merged[col] = merged[col].fillna(1.0)
        else:
            merged[col] = merged[col].fillna(0.0)
    all_std = np.maximum(merged["all_prior_std"].to_numpy(dtype=np.float64), 1e-6)
    season_std = np.maximum(
        merged["season_prior_std"].to_numpy(dtype=np.float64), 1e-6
    )
    for window in RECENT_WINDOWS:
        prefix = f"recent{window}"
        recent_std = np.maximum(
            merged[f"{prefix}_std"].to_numpy(dtype=np.float64), 1e-6
        )
        merged[f"{prefix}_std_to_all"] = recent_std / all_std
        merged[f"{prefix}_std_to_season"] = recent_std / season_std
    return merged


def load_dataset(path):
    df = pd.read_parquet(path)
    df = df[
        df["valid"]
        & df["z_observed"].notna()
        & df["sample_weight"].notna()
        & (df["sample_weight"] > 0.0)
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id", "metric"]).reset_index(drop=True)
    df["metric_code"] = pd.Categorical(df["metric"]).codes.astype(np.float64)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(np.float64)
    df["day_sin"] = np.sin(2.0 * np.pi * df["day_of_year"] / 366.0)
    df["day_cos"] = np.cos(2.0 * np.pi * df["day_of_year"] / 366.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["date"].dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["date"].dt.dayofweek / 7.0)
    season_start = df.groupby("season", observed=True)["date"].transform("min")
    df["season_day"] = (df["date"] - season_start).dt.days.astype(np.float64)
    df["log_sample_weight"] = np.log1p(df["sample_weight"].astype(float))
    df = _add_prior_features(df)

    feature_cols = [
        "metric_code",
        "season",
        "playoffs_fl",
        "season_day",
        "day_sin",
        "day_cos",
        "dow_sin",
        "dow_cos",
        "log_sample_weight",
        *_prior_feature_columns(),
    ]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["z_observed"].to_numpy(dtype=np.float64)
    w = df["sample_weight"].to_numpy(dtype=np.float64)
    return df, X, y, w, feature_cols


def fit_gaussian(name, X_train, y_train, w_train, X_val, y_val, w_val,
                 X_test, y_test, w_test, args, *, sigma_calibration,
                 dist_calibration_feature=None, return_payload=False):
    model_kwargs = dict(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping=True,
        early_stopping_rounds=args.early_stopping_rounds,
        dist_calibration=sigma_calibration,
        eval_metric="nll",
        rho_learning_rate_multiplier=args.rho_learning_rate_multiplier,
        rho_l2_leaf_reg_multiplier=args.rho_l2_leaf_reg_multiplier,
        random_state=args.random_state,
        thread_count=args.thread_count,
        diagnostic_warnings="never",
    )
    if dist_calibration_feature is not None:
        model_kwargs["dist_calibration_feature"] = dist_calibration_feature
    model = ChimeraBoostRegressor(**model_kwargs)
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=[0],
        eval_set=(X_val, y_val),
        sample_weight=w_train,
        eval_sample_weight=w_val,
    )
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mu, sigma = model.predict_dist(X_test)
    predict_seconds = time.perf_counter() - start
    result = _score(
        name,
        X_train.shape[0],
        X_val.shape[0],
        X_test,
        y_test,
        w_test,
        fit_seconds,
        predict_seconds,
        mu,
        sigma,
        best_iteration=model.best_n_estimators_,
        sigma_scale=getattr(model, "sigma_scale_", None),
        sigma_affine_a=getattr(model, "sigma_affine_a_", None),
        sigma_affine_b=getattr(model, "sigma_affine_b_", None),
        bins=args.sigma_bins,
        config=_config_string(args),
    )
    if return_payload:
        return result, {"model": model, "mu": mu, "sigma": sigma}
    return result


def fit_rmse_const_sigma(X_train, y_train, w_train, X_val, y_val, w_val,
                         X_test, y_test, w_test, args):
    model = ChimeraBoostRegressor(
        loss="RMSE",
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping=True,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
        thread_count=args.thread_count,
        diagnostic_warnings="never",
    )
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        cat_features=[0],
        eval_set=(X_val, y_val),
        sample_weight=w_train,
        eval_sample_weight=w_val,
    )
    fit_seconds = time.perf_counter() - start
    val_mu = model.predict(X_val)
    sigma_const = _weighted_rms(y_val - val_mu, w_val)
    start = time.perf_counter()
    mu = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    sigma = np.full_like(mu, max(sigma_const, 1e-12), dtype=np.float64)
    return _score(
        "chimera_rmse_const_sigma",
        X_train.shape[0],
        X_val.shape[0],
        X_test,
        y_test,
        w_test,
        fit_seconds,
        predict_seconds,
        mu,
        sigma,
        best_iteration=model.best_n_estimators_,
        sigma_scale=sigma_const,
        bins=args.sigma_bins,
    )


def score_unit_normal(X_train, X_val, X_test, y_test, w_test, args):
    mu = np.zeros_like(y_test, dtype=np.float64)
    sigma = np.ones_like(y_test, dtype=np.float64)
    return _score(
        "unit_normal_observation_baseline",
        X_train.shape[0],
        X_val.shape[0],
        X_test,
        y_test,
        w_test,
        0.0,
        0.0,
        mu,
        sigma,
        bins=args.sigma_bins,
    )


def _parse_origins(value):
    if value is None or value == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _run_affine_split(
    df, X, y, w, args, *, train_mask, val_mask, test_mask, name
):
    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], w[test_mask]
    result, payload = fit_gaussian(
        name,
        X_train, y_train, w_train, X_val, y_val, w_val,
        X_test, y_test, w_test, args, sigma_calibration="affine",
        return_payload=True,
    )
    frame = _prediction_frame(
        df.loc[test_mask].reset_index(drop=True),
        y_test,
        payload["mu"],
        payload["sigma"],
        w_test,
    )
    return result, payload, frame


def _run_per_metric_affine_split(
    df, X, y, w, args, *, train_mask, val_mask, test_mask
):
    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], w[test_mask]
    result, payload = fit_gaussian(
        "chimera_gaussian_per_metric_affine_calibrated",
        X_train, y_train, w_train, X_val, y_val, w_val,
        X_test, y_test, w_test, args,
        sigma_calibration="per_metric_affine",
        dist_calibration_feature=0,
        return_payload=True,
    )
    frame = _prediction_frame(
        df.loc[test_mask].reset_index(drop=True),
        y_test,
        payload["mu"],
        payload["sigma"],
        w_test,
    )
    return result, payload, frame


def _wnba_tune_candidates(args):
    if not args.tune_gaussian:
        return []
    learning_rates = _parse_grid(args.tune_learning_rates, cast=float) or [
        args.learning_rate
    ]
    num_leaves = _parse_grid(args.tune_num_leaves, cast=int) or [
        args.num_leaves
    ]
    min_child_samples = _parse_grid(
        args.tune_min_child_samples, cast=int
    ) or [args.min_child_samples]
    l2_leaf_regs = _parse_grid(args.tune_l2_leaf_regs, cast=float)
    if not l2_leaf_regs:
        l2_leaf_regs = [args.l2_leaf_reg]
        if str(args.l2_leaf_reg) == "auto":
            l2_leaf_regs.append(1.0)
    rho_multipliers = _parse_grid(
        args.tune_rho_learning_rate_multipliers, cast=float
    ) or [1.0, 0.75, 0.5]
    rho_l2_multipliers = _parse_grid(
        args.tune_rho_l2_leaf_reg_multipliers, cast=float
    ) or [1.0]

    candidates = []
    for lr, leaves, min_child, l2, rho_lr, rho_l2 in product(
        _dedupe(learning_rates),
        _dedupe(num_leaves),
        _dedupe(min_child_samples),
        _dedupe(l2_leaf_regs),
        _dedupe(rho_multipliers),
        _dedupe(rho_l2_multipliers),
    ):
        candidates.append(_with_arg_overrides(
            args,
            learning_rate=float(lr),
            num_leaves=int(leaves),
            min_child_samples=int(min_child),
            l2_leaf_reg=l2,
            rho_learning_rate_multiplier=float(rho_lr),
            rho_l2_leaf_reg_multiplier=float(rho_l2),
        ))
    max_candidates = int(args.tune_max_candidates)
    if max_candidates > 0 and len(candidates) > max_candidates:
        raise ValueError(
            f"WNBA Gaussian tuning grid has {len(candidates)} candidates; "
            f"raise --tune-max-candidates above {max_candidates} or narrow "
            "the grid"
        )
    return candidates


def _run_tuned_affine_split(
    df, X, y, w, args, *, train_mask, val_mask, test_mask
):
    candidates = _wnba_tune_candidates(args)
    if not candidates:
        return None, None, []

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], w[test_mask]
    val_df = df.loc[val_mask].reset_index(drop=True)
    best = None
    rows = []

    for idx, candidate_args in enumerate(candidates, 1):
        result, payload = fit_gaussian(
            f"candidate_{idx}",
            X_train, y_train, w_train, X_val, y_val, w_val,
            X_val, y_val, w_val, candidate_args,
            sigma_calibration="affine",
            return_payload=True,
        )
        frame = _prediction_frame(
            val_df, y_val, payload["mu"], payload["sigma"], w_val
        )
        gate = _gate_stats(_metric_sigma_slice_rows(frame, n_bins=3))
        row = {
            "candidate": int(idx),
            "config": _config_string(candidate_args),
            "val_nll": result.nll,
            "val_crps": result.crps,
            "val_coverage90": result.coverage90,
            "val_std_resid_rms": result.std_resid_rms,
            **gate,
        }
        rows.append(row)
        score = (
            gate["max_metric_rms_dev"],
            gate["mean_metric_rms_dev"],
            result.nll,
            result.crps,
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "args": candidate_args,
                "payload": payload,
                "row": row,
            }

    if best is None:
        return None, None, rows

    model = best["payload"]["model"]
    start = time.perf_counter()
    mu, sigma = model.predict_dist(X_test)
    predict_seconds = time.perf_counter() - start
    selected = _score(
        "chimera_gaussian_affine_tuned",
        X_train.shape[0],
        X_val.shape[0],
        X_test,
        y_test,
        w_test,
        getattr(model.model_, "fit_time_", float("nan")),
        predict_seconds,
        mu,
        sigma,
        best_iteration=model.best_n_estimators_,
        sigma_scale=getattr(model, "sigma_scale_", None),
        sigma_affine_a=getattr(model, "sigma_affine_a_", None),
        sigma_affine_b=getattr(model, "sigma_affine_b_", None),
        bins=args.sigma_bins,
        config=_config_string(best["args"]),
    )
    frame = _prediction_frame(
        df.loc[test_mask].reset_index(drop=True), y_test, mu, sigma, w_test
    )
    best["row"]["selected"] = True
    return selected, frame, rows


def _run_rolling_origins(df, X, y, w, args, origins):
    rows = []
    for origin in origins:
        test_season = origin + 1
        train_mask = df["season"] < origin
        val_mask = df["season"] == origin
        test_mask = df["season"] == test_season
        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            rows.append({
                "origin": origin,
                "test_season": test_season,
                "result": Result(
                    model="chimera_gaussian_affine_calibrated",
                    status="skip",
                    reason="empty rolling-origin split",
                    train_rows=int(train_mask.sum()),
                    val_rows=int(val_mask.sum()),
                    test_rows=int(test_mask.sum()),
                    n_features=int(X.shape[1]),
                ),
            })
            continue
        result, _, _ = _run_affine_split(
            df, X, y, w, args,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            name="chimera_gaussian_affine_calibrated",
        )
        rows.append({
            "origin": origin,
            "test_season": test_season,
            "result": result,
        })
    return rows


def write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(results[0]).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_metric_slices(rows):
    lines = [
        "| Metric | sigma bin | n | 90% cov | std-resid RMS | E[z^2] | mean sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['sigma_bin']} | {row['n']} | "
            f"{_fmt(row['coverage90'])} | {_fmt(row['std_resid_rms'])} | "
            f"{_fmt(row['z2'])} | {_fmt(row['sigma_mean'])} |"
        )
    return lines


def _markdown_lag1(rows):
    lines = [
        "| Group | pairs | lag-1 corr |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['n_pairs']} | {_fmt(row['lag1'])} |"
        )
    return lines


def _markdown_tuning_rows(rows):
    lines = [
        "| selected | candidate | validation NLL | validation CRPS | "
        "90% cov | std-resid RMS | max metric RMS dev | mean metric RMS dev | failed bins | config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {'yes' if row.get('selected') else ''} | {row['candidate']} | "
            f"{_fmt(row['val_nll'])} | {_fmt(row['val_crps'])} | "
            f"{_fmt(row['val_coverage90'])} | {_fmt(row['val_std_resid_rms'])} | "
            f"{_fmt(row['max_metric_rms_dev'])} | "
            f"{_fmt(row['mean_metric_rms_dev'])} | "
            f"{row['failed_metric_bins']} | `{row['config']}` |"
        )
    return lines


def write_markdown(
    results, path, metadata, diagnostics=None, rolling_results=None,
    tuning_rows=None, tuned_diagnostics=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# WNBA Real-Data Distributional Validation",
        "",
        "This is a time-ordered validation on WNBA DARKO game-level metric "
        "observations.  The target is source-column `z_observed`, a "
        "transformed metric observation scale; weights are `sample_weight` "
        "from the observation rows.",
        "",
        "## Data",
        "",
        f"- Source: `{metadata['data_path']}`",
        f"- Date range: {metadata['date_min']} to {metadata['date_max']}",
        f"- Metrics: {', '.join(metadata['metrics'])}",
        f"- Train seasons: {metadata['train_seasons']}",
        f"- Validation seasons: {metadata['val_seasons']}",
        f"- Test seasons: {metadata['test_seasons']}",
        f"- Rows: train {metadata['train_rows']:,}, validation "
        f"{metadata['val_rows']:,}, test {metadata['test_rows']:,}",
        f"- Features: {metadata['n_features']} causal/date/context features; "
        "`metric_code` is categorical.",
        "",
        "## Results",
        "",
        "| Model | NLL | CRPS | RMSE mu | 90% cov | std-resid RMS | "
        "mean sigma | affine b | sigma range | sigma-|resid| corr | fit s | "
        "best iter | config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for result in results:
        sigma_range = f"{_fmt(result.sigma_min)}-{_fmt(result.sigma_max)}"
        lines.append(
            f"| {result.model} | {_fmt(result.nll)} | {_fmt(result.crps)} | "
            f"{_fmt(result.rmse_mu)} | {_fmt(result.coverage90)} | "
            f"{_fmt(result.std_resid_rms)} | {_fmt(result.sigma_mean)} | "
            f"{_fmt(result.sigma_affine_b)} | {sigma_range} | "
            f"{_fmt(result.sigma_abs_resid_corr)} | {_fmt(result.fit_seconds, 2)} | "
            f"{_fmt(result.best_iteration, 0)} | "
            f"{'' if result.config is None else '`' + result.config + '`'} |"
        )
    lines.extend([
        "",
        "## Sigma-Binned Calibration",
        "",
        "Rows are sorted by predicted sigma and split into equal-count bins. "
        "Good observation-noise calibration should keep coverage near 0.90 and "
        "standardized-residual RMS near 1 in each bin.",
        "",
        "| Model | 90% coverage by sigma bin | std-resid RMS by sigma bin | E[z^2] by sigma bin |",
        "|---|---|---|---|",
    ])
    for result in results:
        lines.append(
            f"| {result.model} | {result.coverage90_by_sigma or ''} | "
            f"{result.std_rms_by_sigma or ''} | {result.z2_by_sigma or ''} |"
        )

    if diagnostics:
        lines.extend([
            "",
            "## Affine Calibration Diagnostics",
            "",
            "Diagnostics below use the per-metric affine Gaussian lane when "
            "available, otherwise the affine-calibrated Gaussian lane, on the "
            "held-out test split.",
            "",
            f"- PIT histogram deciles: {diagnostics['pit_histogram']}",
            f"- Weighted PIT KS distance vs uniform: {_fmt(diagnostics['pit_ks'])}",
            "- Lag-1 residual autocorrelation is computed over metric-ordered "
            "game-level rows because this source artifact has no player/entity "
            "identifier; per-player whiteness remains a downstream DARKO replay "
            "gate.",
            "",
            "### Per-Metric Sigma Terciles",
            "",
        ])
        lines.extend(_markdown_metric_slices(diagnostics["metric_sigma_slices"]))
        lines.extend([
            "",
            "### Lag-1 Standardized Residual Correlation",
            "",
        ])
        lines.extend(_markdown_lag1(diagnostics["lag1"]))

    if tuning_rows:
        lines.extend([
            "",
            "## W3 Gaussian Source/Tuning Sweep",
            "",
            "Candidates are selected on the validation fold by worst "
            "per-metric sigma-tercile standardized-residual RMS deviation, "
            "then mean deviation, then NLL/CRPS. The selected row is scored "
            "against the untouched future test seasons in the main results.",
            "",
        ])
        lines.extend(_markdown_tuning_rows(tuning_rows))
        if tuned_diagnostics:
            lines.extend([
                "",
                "### Selected Tuned Lane Per-Metric Sigma Terciles",
                "",
            ])
            lines.extend(
                _markdown_metric_slices(
                    tuned_diagnostics["metric_sigma_slices"]
                )
            )

    if rolling_results:
        lines.extend([
            "",
            "## Rolling-Origin Affine Checks",
            "",
            "Each origin uses seasons before the origin for training, the origin "
            "season for early stopping/calibration, and the next season for "
            "testing.",
            "",
            "| origin val season | test season | NLL | CRPS | 90% cov | std-resid RMS | sigma-bin RMS |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ])
        for item in rolling_results:
            result = item["result"]
            lines.append(
                f"| {item['origin']} | {item['test_season']} | "
                f"{_fmt(result.nll)} | {_fmt(result.crps)} | "
                f"{_fmt(result.coverage90)} | {_fmt(result.std_resid_rms)} | "
                f"{result.std_rms_by_sigma or ''} |"
            )

    calibrated = next(
        (
            r for r in results
            if r.model == "chimera_gaussian_per_metric_affine_calibrated"
        ),
        None,
    )
    if calibrated is None:
        calibrated = next(
            (
                r for r in results
                if r.model == "chimera_gaussian_affine_calibrated"
            ),
            None,
        )
    verdict = "not evaluated"
    if calibrated is not None:
        if (
            calibrated.coverage90 is not None
            and 0.86 <= calibrated.coverage90 <= 0.94
            and calibrated.std_resid_rms is not None
            and 0.90 <= calibrated.std_resid_rms <= 1.10
        ):
            verdict = (
                "passes this real-data scale calibration check, but sigma "
                "still needs downstream Kalman replay validation before use as "
                "production observation noise"
            )
        else:
            verdict = (
                "does not pass this real-data scale calibration check; do not "
                "use sigma as Kalman observation noise without further "
                "calibration"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Calibrated Gaussian verdict: {verdict}.",
        "- This benchmark checks one-step observation calibration on held-out "
        "future seasons; it does not prove Kalman filtering improves when these "
        "sigmas are injected as observation variances.",
        "- The unit-normal baseline is included only as a sanity check for the "
        "source scale; the constant-sigma RMSE lane is the practical "
        "calibration baseline.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=25)
    parser.add_argument("--l2-leaf-reg", default="auto")
    parser.add_argument("--rho-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--rho-l2-leaf-reg-multiplier", type=float, default=1.0)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--train-through-season", type=int, default=2021)
    parser.add_argument("--val-start-season", type=int, default=2022)
    parser.add_argument("--val-through-season", type=int, default=2023)
    parser.add_argument("--sigma-bins", type=int, default=5)
    parser.add_argument(
        "--origins",
        default="",
        help=(
            "comma-separated rolling origins; each origin season is used for "
            "validation/calibration and origin+1 for testing"
        ),
    )
    parser.add_argument(
        "--tune-gaussian",
        action="store_true",
        help="run a small W3 Gaussian tuning sweep and score the selected lane",
    )
    parser.add_argument("--tune-learning-rates", default="")
    parser.add_argument("--tune-num-leaves", default="")
    parser.add_argument("--tune-min-child-samples", default="")
    parser.add_argument("--tune-l2-leaf-regs", default="")
    parser.add_argument("--tune-rho-learning-rate-multipliers", default="")
    parser.add_argument("--tune-rho-l2-leaf-reg-multipliers", default="")
    parser.add_argument("--tune-max-candidates", type=int, default=12)
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise FileNotFoundError(
            f"WNBA observation parquet not found: {args.data}. "
            "Pass --data /path/to/game_metric_observations.parq to run this "
            "private-data benchmark."
        )

    df, X, y, w, feature_cols = load_dataset(args.data)
    train_mask = df["season"] <= args.train_through_season
    val_mask = (
        (df["season"] >= args.val_start_season)
        & (df["season"] <= args.val_through_season)
    )
    test_mask = df["season"] > args.val_through_season
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("empty train/validation/test split")

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], w[test_mask]

    results = [
        score_unit_normal(X_train, X_val, X_test, y_test, w_test, args),
        fit_rmse_const_sigma(
            X_train, y_train, w_train, X_val, y_val, w_val,
            X_test, y_test, w_test, args,
        ),
        fit_gaussian(
            "chimera_gaussian_raw",
            X_train, y_train, w_train, X_val, y_val, w_val,
            X_test, y_test, w_test, args, sigma_calibration=None,
        ),
        fit_gaussian(
            "chimera_gaussian_scalar_calibrated",
            X_train, y_train, w_train, X_val, y_val, w_val,
            X_test, y_test, w_test, args, sigma_calibration="scalar",
        ),
    ]
    affine_result, affine_payload, affine_frame = _run_affine_split(
        df, X, y, w, args,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        name="chimera_gaussian_affine_calibrated",
    )
    results.append(affine_result)
    per_metric_result, per_metric_payload, per_metric_frame = (
        _run_per_metric_affine_split(
            df, X, y, w, args,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    )
    results.append(per_metric_result)
    tuned_result, tuned_frame, tuning_rows = _run_tuned_affine_split(
        df, X, y, w, args,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    if tuned_result is not None:
        results.append(tuned_result)
    diagnostics = _diagnostics_from_frame(per_metric_frame, args.sigma_bins)
    tuned_diagnostics = (
        None if tuned_frame is None
        else _diagnostics_from_frame(tuned_frame, args.sigma_bins)
    )
    rolling_results = _run_rolling_origins(
        df, X, y, w, args, _parse_origins(args.origins)
    )

    metadata = {
        "data_path": str(args.data),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "metrics": sorted(df["metric"].unique().tolist()),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "train_seasons": (
            f"{int(df.loc[train_mask, 'season'].min())}-"
            f"{int(df.loc[train_mask, 'season'].max())}"
        ),
        "val_seasons": (
            f"{int(df.loc[val_mask, 'season'].min())}-"
            f"{int(df.loc[val_mask, 'season'].max())}"
        ),
        "test_seasons": (
            f"{int(df.loc[test_mask, 'season'].min())}-"
            f"{int(df.loc[test_mask, 'season'].max())}"
        ),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "num_leaves": int(args.num_leaves),
        "min_child_samples": int(args.min_child_samples),
        "l2_leaf_reg": args.l2_leaf_reg,
        "rho_learning_rate_multiplier": float(args.rho_learning_rate_multiplier),
        "rho_l2_leaf_reg_multiplier": float(args.rho_l2_leaf_reg_multiplier),
        "thread_count": int(args.thread_count),
        "random_state": int(args.random_state),
        "rolling_origins": _parse_origins(args.origins),
        "tune_gaussian": bool(args.tune_gaussian),
        "lag_diagnostic_group": "metric (source has no player/entity column)",
    }
    write_csv(results, args.csv)
    write_markdown(
        results, args.markdown, metadata,
        diagnostics=diagnostics,
        rolling_results=rolling_results,
        tuning_rows=tuning_rows,
        tuned_diagnostics=tuned_diagnostics,
    )
    print(f"wrote {args.csv}")
    print(f"wrote {args.markdown}")
    for result in results:
        print(
            result.model,
            "nll", _fmt(result.nll),
            "crps", _fmt(result.crps),
            "cov90", _fmt(result.coverage90),
            "std_rms", _fmt(result.std_resid_rms),
            "sigma_mean", _fmt(result.sigma_mean),
        )


if __name__ == "__main__":
    main()
