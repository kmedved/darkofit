# Basketball fresh-worker warmup protocol

## Decision and scope

This phase evaluates an explicit startup warmup for DarkoFit's current
default fit and prediction paths. Basketball is the primary and fatal gate
because it is fast, representative of the project's sports-data priority,
and already has immutable creator-fold, held-team, and cold-player views.
No TabArena or CTR23 task is opened.

The candidate may add only:

- `darkofit.warmup(verbose=False, background=False)`;
- blocking import dispatch through `DARKOFIT_WARMUP=1`; and
- daemon-thread import dispatch through
  `DARKOFIT_WARMUP=background`.

An unset, empty, or zero environment value must do no work. The package's
ordinary import, estimator defaults, fitted state, and predictions must not
change. Warmup time remains startup time outside the measured first
fit/predict; this phase does not claim to eliminate compilation or improve
end-to-end latency when a caller blocks on warmup immediately before one
fit. The intended users are fresh workers that can warm before serving work
or overlap background compilation with other startup tasks.

## Frozen baseline and opportunity

The implementation starts from clean DarkoFit `main` at
`54aef7b0ebc60af4b764719b93eeaf1bcf680f86`. The design comparison is the
clean ChimeraBoost 0.15.0 checkout at
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`. Any substantial literal
adaptation from `chimeraboost/warmup.py` must retain Apache-2.0 attribution
in `NOTICE`.

The exploratory baseline used creator fold 0, 18 threads, the public
`DarkoRegressor(random_state=4)` default, and a unique empty Numba cache:

- first fit: `3.404971375` seconds;
- first prediction: `0.091001` seconds;
- Numba compile dispatch in a separate profile: `3.324` cumulative seconds;
- fitted trees: `1,000`;
- resolved learning rate: `0.052312`; and
- fold prediction SHA-256:
  `6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f`.

The steady-state first-fold fit is approximately 1.5 seconds in prior
basketball campaigns. The frozen opportunity score is `10`:

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score |
|---|---:|---:|---:|---:|
| first-use Numba compilation inside fit/predict | 4 | 5 | 2 | 10 |

`Score = impact * confidence / effort`; this clears the required 2.0 bar.

## Candidate coverage and side effects

The warmup uses deterministic private synthetic data and at most three tiny
fits to compile or load the kernels used by default:

- scalar regression;
- binary and multiclass classification;
- categorical ordered target statistics;
- validation prediction;
- fused multithreaded oblivious-tree construction;
- small-row serial leaf descent; and
- constant-leaf packed prediction.

It deliberately excludes distributional heads, SHAP, local linear leaves,
and non-oblivious tree modes. Those opt-in paths may still compile on first
use. The warmup must restore the caller's Numba thread count, preserve
NumPy's legacy global RNG state, avoid fitted global model state, and avoid
stdout unless `verbose=True`. A background call must return a daemon thread
that terminates successfully.

Focused tests must prove:

- the public function and both environment modes;
- no dispatch for unset, empty, or zero values;
- representative default-path kernels have compiled signatures afterward;
- legacy global RNG and Numba thread-count preservation;
- array-exact deterministic model output before and after warmup;
- background completion; and
- no warmup work on an ordinary package import.

## Frozen fresh-cache basketball campaign

The formal runner uses six reciprocal, position-balanced blocks:

```text
control, warmup
warmup, control
control, warmup
warmup, control
control, warmup
warmup, control
```

Each arm runs in a fresh process with:

- a unique, initially empty `NUMBA_CACHE_DIR`;
- 18 threads;
- `DARKOFIT_WARMUP=0` at import;
- the immutable creator basketball CSV and creator fold 0; and
- the unchanged public `DarkoRegressor(random_state=4)` default.

The control immediately fits and predicts. The candidate calls the public
blocking `warmup()` outside the fit/predict timer, then performs the identical
fit and predictions. Every observation is retained. Cache preparation,
import, warmup, fit, and prediction timings are reported separately. The
runner records fit phase timings, cache-file counts and bytes, fitted
metadata, and predictions for:

- creator fold 0;
- all held-team rows; and
- the corrected 585-row cold-player subset.

The implementation, tests, runner, shared basketball inputs, and this
protocol are content-pinned in the artifact. The runner must itself be
committed and pushed from clean `main == origin/main` before execution, and
the output is create-only.

## Promotion gates

The candidate ships only if all gates pass:

1. Every arm and block reproduces the frozen fold prediction hash and
   array-exact fold, held-team, and cold-player predictions.
2. Every fit resolves to learning rate `0.052312`, 1,000 CatBoost-mode trees,
   and stop reason `iteration_limit`; all timing-free fitted metadata match.
3. Ordinary import performs no warmup. Warmup preserves the tested global
   state and completes in at most 15 seconds per process.
4. Candidate median first-fit time is at most `0.70x` control.
5. Candidate median first-prediction time is at most `0.25x` control.
6. Each arm's first-fit IQR/median is at most `0.25`, and the six paired
   candidate/control fit ratios have IQR/median at most `0.20`.
7. Candidate first-prediction IQR/median is at most `0.50`. Prediction calls
   are only a few milliseconds after warmup, so the ratio gate is primary
   and the absolute spread limit is intentionally wider.
8. No unexpected warning, worker failure, non-finite value, cache reuse, or
   missing cold-player row is permitted.

Failure closes this implementation attempt without changing thresholds or
discarding blocks. Passing authorizes only the explicit warmup API and its
opt-in environment dispatch. It does not authorize a model default, hidden
import work, or a claim that total single-fit cold-start work is lower.
