"""Distributional regression benchmark for ChimeraBoost.

The default run is offline and synthetic. Optional competitors are detected at
runtime and skipped with an explicit reason if their packages are unavailable.

Examples
--------
    python benchmarks/bench_distributional.py
    python benchmarks/bench_distributional.py --datasets synthetic_100k
    python benchmarks/bench_distributional.py --seeds 0 1 2 --n-train 5000
    python benchmarks/bench_distributional.py --models chimera_gaussian ngboost
    python benchmarks/bench_distributional.py --csv /tmp/distributional.csv
    python benchmarks/bench_distributional.py --markdown /tmp/distributional.md
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "chimeraboost-mplconfig"),
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from chimeraboost import ChimeraBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_MODELS = (
    "chimera_gaussian",
    "chimera_gaussian_es",
    "chimera_gaussian_es_calibrated",
    "chimera_rmse_const_sigma",
    "chimera_quantile_pair",
    "ngboost",
    "catboost_uncertainty",
    "lightgbm_twin",
)

OPENML_REGRESSION_DATASETS = {
    "openml_cpu_act": 197,
    "openml_wine_quality": 287,
    "openml_boston": 531,
}


@dataclass
class Result:
    dataset: str
    model: str
    seed: int
    n_train: int
    n_test: int
    n_features: int
    status: str
    reason: str = ""
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    best_iteration: int | None = None
    rmse_mu: float | None = None
    nll: float | None = None
    crps: float | None = None
    interval90_coverage: float | None = None
    interval90_width: float | None = None
    cov90_by_sigma: str | None = None
    sigma_mean: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def _make_heteroscedastic(seed, n_train, n_test, n_features):
    if n_features < 6:
        raise ValueError("n_features must be at least 6 for this benchmark")
    rng = np.random.default_rng(seed)
    n = int(n_train) + int(n_test)
    X = rng.normal(size=(n, n_features))
    sigma = (
        0.20
        + 0.65 / (1.0 + np.exp(-2.0 * X[:, 1]))
        + 0.20 * np.abs(X[:, 3])
    )
    mu = (
        1.4 * X[:, 0]
        - 0.8 * X[:, 2]
        + 0.4 * np.sin(2.0 * X[:, 4])
        + 0.2 * X[:, 5] * X[:, 0]
    )
    y = mu + rng.normal(scale=sigma)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _synthetic_dataset(dataset, seed, args):
    if dataset == "synthetic_smoke":
        return _make_heteroscedastic(seed, args.n_train, args.n_test, args.n_features)
    if dataset == "synthetic_100k":
        return _make_heteroscedastic(seed, 100_000, 25_000, args.n_features)
    if dataset == "synthetic_500k":
        return _make_heteroscedastic(seed, 500_000, 100_000, args.n_features)
    raise KeyError(dataset)


def _openml_dataset(dataset, seed):
    from sklearn.datasets import fetch_openml

    data_id = OPENML_REGRESSION_DATASETS[dataset]
    ds = fetch_openml(data_id=data_id, as_frame=True)
    target = ds.target.astype(float).to_numpy()
    X_df = ds.frame.drop(columns=[ds.target.name])
    X_df = X_df.apply(lambda col: col.astype(float))
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.to_numpy(dtype=np.float64)
    return train_test_split(X, target, test_size=0.25, random_state=seed)


def _make_dataset(dataset, seed, args):
    if dataset.startswith("synthetic_"):
        return _synthetic_dataset(dataset, seed, args)
    if dataset in OPENML_REGRESSION_DATASETS:
        return _openml_dataset(dataset, seed)
    raise KeyError(dataset)


def _normal_nll(y, mu, sigma):
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    z = (np.asarray(y, dtype=np.float64) - mu) / sigma
    return float(np.mean(0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z * z))


def _normal_crps(y, mu, sigma):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    return float(
        np.mean(
            sigma
            * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        )
    )


def _sigma_binned_coverage(y, lo, hi, sigma, n_bins):
    n_bins = int(n_bins)
    if n_bins < 2:
        return None
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 0:
        return None
    order = np.argsort(sigma)
    covered = ((np.asarray(y) >= lo) & (np.asarray(y) <= hi))[order]
    chunks = [chunk for chunk in np.array_split(covered, n_bins) if chunk.size]
    return "/".join(f"{float(np.mean(chunk)):.3f}" for chunk in chunks)


def _score(dataset, model_name, seed, Xtr, Xte, yte, fit_seconds, predict_seconds,
           mu, sigma, best_iteration=None, sigma_bins=5):
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-12, None)
    mu = np.asarray(mu, dtype=np.float64)
    lo = mu - 1.6448536269514722 * sigma
    hi = mu + 1.6448536269514722 * sigma
    return Result(
        dataset=dataset,
        model=model_name,
        seed=int(seed),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        n_features=int(Xtr.shape[1]),
        status="ok",
        fit_seconds=float(fit_seconds),
        predict_seconds=float(predict_seconds),
        best_iteration=None if best_iteration is None else int(best_iteration),
        rmse_mu=float(np.sqrt(np.mean((yte - mu) ** 2))),
        nll=_normal_nll(yte, mu, sigma),
        crps=_normal_crps(yte, mu, sigma),
        interval90_coverage=float(np.mean((yte >= lo) & (yte <= hi))),
        interval90_width=float(np.mean(hi - lo)),
        cov90_by_sigma=_sigma_binned_coverage(
            yte, lo, hi, sigma, sigma_bins
        ),
        sigma_mean=float(np.mean(sigma)),
        sigma_min=float(np.min(sigma)),
        sigma_max=float(np.max(sigma)),
    )


def _score_interval_only(dataset, model_name, seed, Xtr, Xte, yte,
                         fit_seconds, predict_seconds, lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    mu = 0.5 * (lo + hi)
    return Result(
        dataset=dataset,
        model=model_name,
        seed=int(seed),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        n_features=int(Xtr.shape[1]),
        status="ok",
        reason="interval-only baseline",
        fit_seconds=float(fit_seconds),
        predict_seconds=float(predict_seconds),
        rmse_mu=float(np.sqrt(np.mean((yte - mu) ** 2))),
        interval90_coverage=float(np.mean((yte >= lo) & (yte <= hi))),
        interval90_width=float(np.mean(hi - lo)),
    )


def _skip(dataset, model_name, seed, Xtr, Xte, reason):
    return Result(
        dataset=dataset,
        model=model_name,
        seed=int(seed),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        n_features=int(Xtr.shape[1]),
        status="skip",
        reason=reason,
    )


def _run_chimera_gaussian(dataset, seed, Xtr, Xte, ytr, yte, args):
    t0 = time.perf_counter()
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        thread_count=args.threads,
        random_state=seed,
        diagnostic_warnings="never",
    ).fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    mu, sigma = model.predict_dist(Xte)
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "chimera_gaussian", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        best_iteration=model.n_estimators_,
        sigma_bins=args.sigma_bins,
    )


def _run_chimera_gaussian_es(dataset, seed, Xtr, Xte, ytr, yte, args):
    t0 = time.perf_counter()
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=(
            args.early_stop_iterations
            if args.early_stop_iterations is not None
            else args.iterations
        ),
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        early_stopping=True,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
        thread_count=args.threads,
        random_state=seed,
        diagnostic_warnings="never",
    ).fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    mu, sigma = model.predict_dist(Xte)
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "chimera_gaussian_es", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        best_iteration=model.n_estimators_,
        sigma_bins=args.sigma_bins,
    )


def _run_chimera_gaussian_es_calibrated(dataset, seed, Xtr, Xte, ytr, yte,
                                        args):
    t0 = time.perf_counter()
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=(
            args.early_stop_iterations
            if args.early_stop_iterations is not None
            else args.iterations
        ),
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        early_stopping=True,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
        sigma_calibration="scalar",
        thread_count=args.threads,
        random_state=seed,
        diagnostic_warnings="never",
    ).fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    mu, sigma = model.predict_dist(Xte)
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "chimera_gaussian_es_calibrated", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        best_iteration=model.n_estimators_,
        sigma_bins=args.sigma_bins,
    )


def _run_chimera_rmse_const_sigma(dataset, seed, Xtr, Xte, ytr, yte, args):
    t0 = time.perf_counter()
    model = ChimeraBoostRegressor(
        loss="RMSE",
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        thread_count=args.threads,
        random_state=seed,
        diagnostic_warnings="never",
    ).fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    train_mu = model.predict(Xtr)
    sigma_const = float(np.sqrt(np.mean((ytr - train_mu) ** 2)))
    t0 = time.perf_counter()
    mu = model.predict(Xte)
    sigma = np.full_like(mu, sigma_const, dtype=np.float64)
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "chimera_rmse_const_sigma", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        best_iteration=model.n_estimators_,
        sigma_bins=args.sigma_bins,
    )


def _run_chimera_quantile_pair(dataset, seed, Xtr, Xte, ytr, yte, args):
    t0 = time.perf_counter()
    common = dict(
        tree_mode="lightgbm",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        thread_count=args.threads,
        random_state=seed,
        diagnostic_warnings="never",
    )
    lo_model = ChimeraBoostRegressor(loss="Quantile", alpha=0.05, **common)
    hi_model = ChimeraBoostRegressor(loss="Quantile", alpha=0.95, **common)
    lo_model.fit(Xtr, ytr)
    hi_model.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    lo = lo_model.predict(Xte)
    hi = hi_model.predict(Xte)
    predict_seconds = time.perf_counter() - t0
    return _score_interval_only(
        dataset, "chimera_quantile_pair", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, np.minimum(lo, hi), np.maximum(lo, hi),
    )


def _run_ngboost(dataset, seed, Xtr, Xte, ytr, yte, args):
    if not _has_module("ngboost"):
        return _skip(dataset, "ngboost", seed, Xtr, Xte, "ngboost is not installed")
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore

    t0 = time.perf_counter()
    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        minibatch_frac=1.0,
        col_sample=1.0,
        random_state=seed,
        verbose=False,
    )
    model.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    dist = model.pred_dist(Xte)
    params = getattr(dist, "params", {})
    mu = getattr(dist, "loc", params.get("loc"))
    sigma = getattr(dist, "scale", params.get("scale"))
    if mu is None or sigma is None:
        raise RuntimeError("unexpected NGBoost Normal distribution payload")
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "ngboost", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        sigma_bins=args.sigma_bins,
    )


def _run_catboost_uncertainty(dataset, seed, Xtr, Xte, ytr, yte, args):
    if not _has_module("catboost"):
        return _skip(
            dataset, "catboost_uncertainty", seed, Xtr, Xte,
            "catboost is not installed"
        )
    from catboost import CatBoostRegressor

    t0 = time.perf_counter()
    model = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=6,
        random_seed=seed,
        thread_count=args.threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    raw = np.asarray(model.predict(Xte, prediction_type="RawFormulaVal"))
    if raw.ndim != 2:
        raise RuntimeError(f"unexpected CatBoost prediction shape {raw.shape}")
    if raw.shape == (2, Xte.shape[0]):
        raw = raw.T
    if raw.shape != (Xte.shape[0], 2):
        raise RuntimeError(f"unexpected CatBoost prediction shape {raw.shape}")
    mu = raw[:, 0]
    sigma = np.exp(np.clip(raw[:, 1], -15.0, 15.0))
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "catboost_uncertainty", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        best_iteration=getattr(model, "tree_count_", None),
        sigma_bins=args.sigma_bins,
    )


def _lightgbm_regressor(seed, args):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        random_state=seed,
        n_jobs=args.threads,
        verbosity=-1,
    )


def _run_lightgbm_twin(dataset, seed, Xtr, Xte, ytr, yte, args):
    if not _has_module("lightgbm"):
        return _skip(
            dataset, "lightgbm_twin", seed, Xtr, Xte, "lightgbm is not installed"
        )

    t0 = time.perf_counter()
    oof = np.empty_like(ytr, dtype=np.float64)
    splitter = KFold(n_splits=args.lightgbm_oof_folds, shuffle=True,
                     random_state=seed)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(Xtr)):
        fold_model = _lightgbm_regressor(seed + fold + 1, args)
        fold_model.fit(Xtr[fit_idx], ytr[fit_idx])
        oof[val_idx] = fold_model.predict(Xtr[val_idx])

    mean_model = _lightgbm_regressor(seed, args)
    mean_model.fit(Xtr, ytr)
    log_var_target = np.log((ytr - oof) ** 2 + args.lightgbm_variance_eps)
    var_model = _lightgbm_regressor(seed + 10_000, args)
    var_model.fit(Xtr, log_var_target)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    mu = mean_model.predict(Xte)
    sigma = np.sqrt(
        np.exp(np.clip(var_model.predict(Xte), -30.0, 30.0))
    )
    predict_seconds = time.perf_counter() - t0
    return _score(
        dataset, "lightgbm_twin", seed, Xtr, Xte, yte,
        fit_seconds, predict_seconds, mu, sigma,
        sigma_bins=args.sigma_bins,
    )


RUNNERS = {
    "chimera_gaussian": _run_chimera_gaussian,
    "chimera_gaussian_es": _run_chimera_gaussian_es,
    "chimera_gaussian_es_calibrated": _run_chimera_gaussian_es_calibrated,
    "chimera_rmse_const_sigma": _run_chimera_rmse_const_sigma,
    "chimera_quantile_pair": _run_chimera_quantile_pair,
    "ngboost": _run_ngboost,
    "catboost_uncertainty": _run_catboost_uncertainty,
    "lightgbm_twin": _run_lightgbm_twin,
}


def _fmt(value, digits=5):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_results(rows):
    print("\nDistributional regression benchmark")
    model_width = max([34, *(len(r.model) for r in rows)])
    print(
        f"{'dataset':18s} {'model':{model_width}s} {'seed':>4s} {'status':>6s} {'fit_s':>8s} "
        f"{'pred_s':>8s} {'rmse_mu':>9s} {'nll':>9s} {'crps':>9s} "
        f"{'cov90':>7s} {'width90':>8s} {'cov90_by_sigma':>29s} "
        f"{'sigma':>20s} reason"
    )
    for r in rows:
        sigma = "-"
        if r.sigma_mean is not None:
            sigma = (
                f"{r.sigma_mean:.4f}"
                f"[{r.sigma_min:.4f},{r.sigma_max:.4f}]"
            )
        print(
            f"{r.dataset:18s} {r.model:{model_width}s} {r.seed:4d} {r.status:>6s} "
            f"{_fmt(r.fit_seconds, 3):>8s} {_fmt(r.predict_seconds, 3):>8s} "
            f"{_fmt(r.rmse_mu):>9s} {_fmt(r.nll):>9s} "
            f"{_fmt(r.crps):>9s} {_fmt(r.interval90_coverage, 3):>7s} "
            f"{_fmt(r.interval90_width, 3):>8s} "
            f"{_fmt(r.cov90_by_sigma):>29s} {sigma:>20s} {r.reason}"
        )

    ok_rows = [r for r in rows if r.status == "ok"]
    if ok_rows:
        print("\nAverages over successful runs")
        for dataset, model in sorted({(r.dataset, r.model) for r in ok_rows}):
            subset = [
                r for r in ok_rows if r.dataset == dataset and r.model == model
            ]
            nll_values = [r.nll for r in subset if r.nll is not None]
            crps_values = [r.crps for r in subset if r.crps is not None]
            print(
                f"  {dataset:18s} {model:{model_width}s}"
                f" fit={statistics.mean(r.fit_seconds for r in subset):.3f}s"
                f" nll={_fmt(statistics.mean(nll_values) if nll_values else None)}"
                f" crps={_fmt(statistics.mean(crps_values) if crps_values else None)}"
                f" cov90={statistics.mean(r.interval90_coverage for r in subset):.3f}"
                f" width90={statistics.mean(r.interval90_width for r in subset):.3f}"
            )
    print("\nMarkdown table")
    print(_markdown_table(rows))


def _markdown_table(rows):
    headers = [
        "dataset", "model", "seed", "status", "fit_s", "rmse_mu", "nll",
        "crps", "cov90", "width90", "cov90_by_sigma", "reason",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        values = [
            r.dataset,
            r.model,
            str(r.seed),
            r.status,
            _fmt(r.fit_seconds, 3),
            _fmt(r.rmse_mu),
            _fmt(r.nll),
            _fmt(r.crps),
            _fmt(r.interval90_coverage, 3),
            _fmt(r.interval90_width, 3),
            _fmt(r.cov90_by_sigma),
            r.reason,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_table(rows) + "\n", encoding="utf-8")


def _warm_chimera(args):
    if args.no_warmup:
        return
    Xtr, Xte, ytr, yte = _make_heteroscedastic(
        seed=1729,
        n_train=128,
        n_test=32,
        n_features=max(6, args.n_features),
    )
    warm_args = argparse.Namespace(**vars(args))
    warm_args.iterations = min(max(1, int(args.iterations)), 2)
    if warm_args.early_stop_iterations is not None:
        warm_args.early_stop_iterations = min(
            max(1, int(warm_args.early_stop_iterations)), 2
        )
    warm_args.num_leaves = min(max(3, int(args.num_leaves)), 7)
    warm_args.min_child_samples = min(max(2, int(args.min_child_samples)), 4)
    if "chimera_gaussian" in args.models:
        _run_chimera_gaussian("warmup", 0, Xtr, Xte, ytr, yte, warm_args)
    if "chimera_gaussian_es" in args.models:
        _run_chimera_gaussian_es("warmup", 0, Xtr, Xte, ytr, yte, warm_args)
    if "chimera_gaussian_es_calibrated" in args.models:
        _run_chimera_gaussian_es_calibrated(
            "warmup", 0, Xtr, Xte, ytr, yte, warm_args
        )
    if "chimera_rmse_const_sigma" in args.models:
        _run_chimera_rmse_const_sigma("warmup", 0, Xtr, Xte, ytr, yte, warm_args)
    if "chimera_quantile_pair" in args.models:
        _run_chimera_quantile_pair("warmup", 0, Xtr, Xte, ytr, yte, warm_args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_choices = [
        "synthetic_smoke", "synthetic_100k", "synthetic_500k",
        *sorted(OPENML_REGRESSION_DATASETS),
    ]
    parser.add_argument("--datasets", nargs="+", choices=dataset_choices,
                        default=["synthetic_smoke"])
    parser.add_argument("--models", nargs="+", choices=sorted(RUNNERS),
                        default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--early-stop-iterations",
        type=int,
        help="max rounds for chimera_gaussian_es; defaults to --iterations",
    )
    parser.add_argument("--early-stopping-rounds", default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sigma-bins", type=int, default=5)
    parser.add_argument("--lightgbm-oof-folds", type=int, default=3)
    parser.add_argument("--lightgbm-variance-eps", type=float, default=1e-6)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--no-warmup", action="store_true",
                        help="include first-call ChimeraBoost JIT time")
    args = parser.parse_args(argv)
    if args.n_features < 6:
        parser.error("--n-features must be at least 6")
    if args.sigma_bins < 1:
        parser.error("--sigma-bins must be at least 1")
    if args.early_stop_iterations is not None and args.early_stop_iterations < 1:
        parser.error("--early-stop-iterations must be positive")
    if args.early_stopping_rounds != "auto":
        try:
            args.early_stopping_rounds = int(args.early_stopping_rounds)
        except ValueError as exc:
            raise SystemExit(
                "--early-stopping-rounds must be an integer or 'auto'"
            ) from exc
        if args.early_stopping_rounds < 1:
            parser.error("--early-stopping-rounds must be positive")
    if args.lightgbm_oof_folds < 2:
        parser.error("--lightgbm-oof-folds must be at least 2")
    return args


def main(argv=None):
    args = parse_args(argv)
    _warm_chimera(args)
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                Xtr, Xte, ytr, yte = _make_dataset(dataset, seed, args)
            except Exception as exc:
                for model_name in args.models:
                    rows.append(Result(
                        dataset=dataset,
                        model=model_name,
                        seed=int(seed),
                        n_train=0,
                        n_test=0,
                        n_features=0,
                        status="skip",
                        reason=f"dataset load failed: {exc}",
                    ))
                continue
            for model_name in args.models:
                rows.append(
                    RUNNERS[model_name](dataset, seed, Xtr, Xte, ytr, yte, args)
                )
    _print_results(rows)
    if args.csv is not None:
        _write_csv(args.csv, rows)
        print(f"\nWrote CSV: {args.csv}")
    if args.markdown is not None:
        _write_markdown(args.markdown, rows)
        print(f"Wrote markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
