# Basketball diagnostic against ChimeraBoost 0.18

_Run on 2026-07-19. This is descriptive Tier-E characterization on already
spent basketball data. It is not a preregistered confirmation, does not amend
an older frozen result, and did not access the CTR23 lockbox._

## Outcome

ChimeraBoost 0.18 did not overturn DarkoFit's sports-quality advantage.
DarkoFit and ChimeraBoost were effectively tied on the creator's
overlap-permitting ten-fold score, with a small DarkoFit advantage under the
actual ChimeraBoost 0.18 default. On the stronger player-disjoint panel,
DarkoFit retained lower primary and cold-player RMSE, while ChimeraBoost
retained substantially lower fit and prediction time.

ChimeraBoost's new default-on quantized-gradient histograms were mildly harmful
on this small, noisy sports workload. Disabling quantization restored the
ChimeraBoost 0.15 player-disjoint quality values exactly to the printed
precision. The updated eight-member ChimeraBoost ensemble achieved the highest
creator-fold R2 in this diagnostic, at materially higher cost.

No DarkoFit default change is authorized by this diagnostic.

## Source and environment

| Item | Bound value |
| --- | --- |
| DarkoFit source | clean `main`, `ec66a64654becaf948592588a047bfb8205decc8`, tag `v0.10.0` |
| ChimeraBoost source | clean `main`, `f14be606b641f1bf0dc92bb14b3951f1fe631c6b` |
| ChimeraBoost description | `v0.18.0-6-gf14be60`; local `main`, `origin/main`, and `upstream/main` agreed |
| Machine | Apple M5 Max, arm64, 18 logical CPUs |
| Python | 3.12.13 |
| NumPy / Numba | 2.4.6 / 0.65.1 |
| pandas / scikit-learn | 3.0.3 / 1.9.0 |
| CatBoost wheel present | 1.2.10; CatBoost was not rerun in this diagnostic |
| Thread policy | 18 threads; warmup completed outside each measured worker |
| Model seed | 4 |

The ChimeraBoost checkout contained the tagged 0.18.0 release plus six audit
commits. The material 0.18 changes were default-on quantized-gradient
histograms and a fused grow-level kernel. The intervening 0.16 releases also
changed bagged-member sampling, automatic member parameters, preprocessing
reuse, and ensemble parallelism.

## Data and scoring boundaries

### Creator-fold view

The run reused the exact data and transform in
[`basketball_creator_benchmark_protocol.md`](basketball_creator_benchmark_protocol.md):

- raw CSV SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`;
- processed training-feature SHA-256:
  `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`;
- processed target SHA-256:
  `7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf`;
- unweighted, unshuffled ten-fold `KFold`;
- mean held-fold R2;
- folds evaluated sequentially after one complete first-fold warmup per arm.

This view is useful because it reproduces the creator's benchmark shape, but it
is not player-disjoint. Players can appear in both training and validation
folds, so this is not the binding sports generalization view.

### Player-disjoint and cold-player view

The run reused the spent 2014–2016 panel and split plan from
[`basketball_sports_panel_v2_protocol.md`](basketball_sports_panel_v2_protocol.md):

- exact 204 MB raw-source SHA-256:
  `96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826`;
- rebuilt processed-panel SHA-256:
  `8f7eab3765b4166740b150ed372f9607bcd6dd9673e0e73cc6541583230a59e6`;
- processed identity SHA-256:
  `ce2e2c9601994479ad14ec9a5cf3068b68b4564c9f341fe4ec8955dc23f3da46`;
- three seasons by three targets, for nine equally weighted lineages;
- ten player-disjoint `GroupKFold` folds per lineage;
- separate held-team, seen-player, and 585-row cold-player views.

The derived panel cache had been cleaned locally. It was rebuilt with the
committed builder from the still-present exact raw source. The rebuilt bytes
matched the frozen manifest before any model ran. This was reconstruction of
spent input data, not a new data choice or a new confirmation panel.

## Creator-fold results

Higher R2 is better. Default-arm wall times are medians across three fresh
workers. Diagnostic-arm timings are single warmed runs and are labeled
accordingly.

| Arm | Quantized gradients | Members | Mean R2 | Measured wall | Repeats |
| --- | :---: | ---: | ---: | ---: | ---: |
| DarkoFit 0.10 default | n/a | 1 | **0.526749518388** | **9.7329s median** | 3 |
| ChimeraBoost 0.18 default | yes | 1 | 0.525992833587 | 12.9211s median | 3 |
| ChimeraBoost 0.18 float diagnostic | no | 1 | **0.526981194352** | 11.6885s | 1 |
| ChimeraBoost 0.18 ensemble 5 | yes | 5 | 0.541906143164 | 36.8584s | 1 |
| ChimeraBoost 0.18 ensemble 8 | yes | 8 | **0.543327284448** | 54.7830s | 1 |

The three default repeats produced identical fold scores for each model.
Their measured wall series were:

- DarkoFit: `10.9801`, `9.6852`, `9.7329` seconds;
- ChimeraBoost: `12.9211`, `12.0409`, `13.0936` seconds.

On this exact workload, DarkoFit's median wall time was 24.70% lower. The
actual single-model scores differed by only `0.000756685` R2 in DarkoFit's
favor and are best read as an effective tie, not a general superiority claim.

The single-model float ChimeraBoost diagnostic differed from its quantized
default on only the tenth creator fold. The first nine fold scores were exact
ties; the final fold was `0.407212360017` without quantization versus
`0.397328752370` with quantization. This one fold accounts for the aggregate
quality difference between those ChimeraBoost lanes.

For completeness, the ten creator-fold R2 values were:

| Arm | Fold R2 values |
| --- | --- |
| DarkoFit default | `0.567679, 0.605376, 0.561310, 0.621107, 0.559958, 0.566401, 0.536777, 0.415441, 0.409577, 0.423870` |
| ChimeraBoost default | `0.564317, 0.588450, 0.565182, 0.638570, 0.568560, 0.569387, 0.539895, 0.412037, 0.416202, 0.397329` |
| ChimeraBoost ensemble 5 | `0.570417, 0.605446, 0.574331, 0.641948, 0.579807, 0.590617, 0.549580, 0.433254, 0.433060, 0.440602` |
| ChimeraBoost ensemble 8 | `0.571898, 0.610601, 0.574406, 0.639530, 0.575832, 0.595749, 0.548107, 0.439211, 0.436740, 0.441201` |

The ensemble result is an accuracy/cost option, not an apples-to-apples
single-default win. The eight-member arm took 5.63 times DarkoFit's
single-model median wall time.

## Player-disjoint results

Lower RMSE is better. The ChimeraBoost 0.18 default timing is the median of
three deterministic repeats. The DarkoFit and float-ChimeraBoost diagnostics
were each run once after an out-of-timing warmup.

| Arm | Primary RMSE | Held-team RMSE | Cold-player RMSE | Fit | Predict | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DarkoFit 0.10 default | **1.963256746236** | **1.795738914502** | **1.807505757021** | 95.9875s | 0.08780s | 1 |
| ChimeraBoost 0.18 default | 2.021610226388 | 1.829339745649 | 1.841096526736 | **34.7151s median** | **0.06189s median** | 3 |
| ChimeraBoost 0.18 float diagnostic | 2.020069634615 | 1.828681531995 | 1.840381025460 | 31.0870s | 0.06369s | 1 |

The ChimeraBoost default produced the same behavior fingerprint in all three
repeats:
`3e582ba5c586d54cf4af5efca5d5fe24713e621dd21dd1d48b9cea24b0557a85`.

Relative to the actual ChimeraBoost 0.18 default:

- DarkoFit's equal-lineage primary RMSE was 2.89% lower;
- DarkoFit's cold-player RMSE was 1.82% lower;
- ChimeraBoost fit 2.77 times faster; and
- ChimeraBoost predicted 1.42 times faster.

Relative to the float ChimeraBoost lane, default quantization increased
primary RMSE by 0.0763% and cold-player RMSE by 0.0389%. The float lane's
primary and cold-player aggregates exactly reproduce the published
ChimeraBoost 0.15 values in
[`basketball_sports_panel_v2_result.md`](basketball_sports_panel_v2_result.md).
That isolates quantized split selection as the cause of the 0.18 quality
movement on this panel.

The quantized lane was also slower than the one float diagnostic on this
small-data workload. That observation does not contradict ChimeraBoost's
reported larger-suite speedups: fixed quantization overhead can dominate these
small fits, and the float timing has only one repeat. It does mean the broad
speed claim should not be imported into a basketball-scale claim.

## Execution record

The default creator arms used the existing runner with the source-drift flag
explicitly enabled so the old baseline could not be mistaken for a frozen
baseline-eligible rerun:

```bash
python benchmarks/run_basketball_creator_benchmark.py \
  --lane steady \
  --arms darkofit_default chimeraboost_default \
  --allow-chimeraboost-drift \
  --output /private/tmp/basketball_creator_chimera_v018_steady_N.json
```

The first repeat also included `chimeraboost_ensemble5`. The float and
eight-member diagnostics reused the runner's exact loader, transform, folds,
warmup, and sequential scoring while changing only
`quantize_gradients=False` or `n_ensembles=8`, respectively.

The player-disjoint cache was rebuilt with:

```bash
PYTHONPATH=. python benchmarks/build_basketball_sports_panel_v2.py \
  --manifest /private/tmp/basketball_sports_panel_v2_rebuilt_manifest.json
```

The committed sports-panel worker was then called directly on the existing
`darkofit_control` and ChimeraBoost comparator routes. For the float
diagnostic, the ChimeraBoost estimator changed only
`quantize_gradients=False`.

## Evidence limitations

- This work used already-spent basketball data and is descriptive Tier-E
  characterization. It cannot promote or reject a DarkoFit default.
- It was not source-frozen prospectively. The exact revisions, environment,
  inputs, parameters, fold scores, repeat timings, and behavior fingerprint
  are recorded here, but the interactive float/ensemble and sports summaries
  are not create-only raw campaign artifacts.
- The creator view permits player overlap. The player-disjoint and cold-player
  views are the stronger sports evidence.
- CatBoost was not rerun.
- The 13-dataset regression panel was not rerun against ChimeraBoost 0.18.
  The older broad-tabular result therefore remains historically valid for its
  bound versions, but is not a current-version comparison.
- One-run timings are descriptive only. Only the three-repeat default-arm
  medians should be used for timing comparisons from this diagnostic.

## Disposition

1. Keep DarkoFit's noisy-data and sports defaults unchanged.
2. Do not copy a default-on quantized-gradient policy from this evidence.
3. Treat ChimeraBoost's improved eight-member ensemble as the material new
   accuracy competitor on the creator-fold workload, with its cost shown
   alongside the score.
4. If broad default comparison becomes the next priority, rerun the frozen
   13-dataset design against ChimeraBoost 0.18 on the same machine rather than
   extrapolating from either basketball or ChimeraBoost's own benchmark suite.
