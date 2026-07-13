# TabArena scalar-regression cap-horizon protocol

Status: **frozen before execution**. This protocol tests one causal question:
with every other model setting held fixed, does raising DarkoFit's maximum
boosting horizon from 1,000 to 10,000 rounds improve generalization when early
stopping and best-prefix retention are enabled?

This is not a default-policy sweep. `tree_mode="auto"`, four target-statistic
permutations, alternate categorical representations, and `linear_residual` are
separate experiments. They must not be added to this run.

Schema preflight disclosure: before the full campaign, one `QSAR_fish_toxicity`
`r0f0` pair was run with a 600-second development budget solely to exercise the
live eight-child result schema. Both arms early-stopped by round 289 and had
identical RMSE. That score was not used to choose an arm, threshold, dataset, or
split; the production campaign starts from a separate zero-start cache at the
frozen 3,600-second budget. Subsequent review fixes only tightened artifact and
analysis provenance, warmup branch coverage, enforcement of already declared
settings, and explicit reporting of the pre-existing one-sided decision
statistic.

## Frozen arms

Both arms use the same outer TabArena splits, bag folds, fold-wise seeds,
resource budget, validation data, and parameters below. The outer-job budget is
frozen at **3,600 seconds**. The only arm difference is `iterations`.

| Parameter | 1,000-round arm | 10,000-round arm |
|---|---:|---:|
| `iterations` | 1,000 | 10,000 |
| `tree_mode` | `"catboost"` | `"catboost"` |
| `l2_leaf_reg` | 3 | 3 |
| `max_bins` | 128 | 128 |
| `learning_rate` | 0.1 | 0.1 |
| `ts_permutations` | 1 | 1 |
| `early_stopping` | `True` | `True` |
| `use_best_model` | `True` | `True` |

The fixed learning rate is essential. Allowing automatic learning rate would
change learning rate when the cap changes, confounding horizon with step size.
L2 remains at 3 to avoid reintroducing the known Concrete-specific L2 issue.

Each TabArena outer job fits eight bag children with one bag set,
`model_random_seed=0`, `vary_seed_across_folds=True`, and sequential local fold
fitting. The adapter's soft monotonic deadline uses the remaining time budget
that AutoGluon passes to each child. A tree already in progress may finish
after that deadline; fitted metadata must record the stop reason and elapsed
deadline state.

## Frozen 13-dataset panel

The panel contains every coordinate shared by the original 13-dataset
regression comparison. There are 222 dataset/split coordinates and two arms,
for **444 outer jobs and 3,552 expected child fits**.

| Dataset | Task ID | Splits |
|---|---:|---:|
| Airfoil self noise | 363612 | 30 |
| Used Fiat 500 | 363615 | 30 |
| Concrete compressive strength | 363625 | 30 |
| Diamonds | 363631 | 9 |
| Food delivery time | 363672 | 9 |
| Healthcare insurance expenses | 363675 | 30 |
| Houses | 363678 | 9 |
| Miami housing | 363686 | 9 |
| Physiochemical protein | 363693 | 9 |
| QSAR-TID-11 | 363697 | 9 |
| QSAR fish toxicity | 363698 | 30 |
| Superconductivity | 363705 | 9 |
| Wine quality | 363708 | 9 |
| **Total** |  | **222** |

A 30-split task uses `r0f0` through `r9f2`; a nine-split task uses `r0f0`
through `r2f2`. The runner requests the superset `r0f0` through `r9f2`, then
requires the built jobs to match the exact sparse coordinate map above.

## Execution order and warmup

The two jobs for each task/split coordinate run adjacent to reduce machine-state
drift. The 1,000-round arm runs first on even-numbered coordinates; the
10,000-round arm runs first on odd-numbered coordinates. This balanced reversal
prevents a persistent first- or second-run advantage from aligning with one
arm. The runner reconstructs this order explicitly and rejects missing,
duplicate, or unexpected jobs.

Before `TabArenaContext.run_jobs`, the process runs the benchmark-local numeric
and categorical regression warmup. Each lane predicts immediately below and
at the production flat-prediction parallel threshold, compiling both the
serial and parallel kernels when multiple threads are available. Its JIT/cache
cost is therefore outside the measured campaign. Warmup routes, shapes,
duration, resolved settings, and deterministic prediction fingerprints are
appended to `warmup_history.json` for each fresh or resumed process. Before
warmup, the runner resolves AutoGluon's sequential child CPU allocation from
the built experiments, pins that value explicitly on both experiment arms,
and passes the identical value as DarkoFit's warmup `thread_count`. Completion
requires every fitted child's recorded CPU allocation and every warmup stage's
resolved thread count to equal that manifest-bound value.

## Zero-start and provenance rules

The default is a zero-start run:

- The output directory must be missing or empty.
- The entire DarkoFit repository and the imported editable TabArena checkout
  must be clean and committed. The runner also proves that the imported
  `darkofit` module resolves to this repository.
- Before warmup or model fitting, `run_manifest.json` binds the cache to the
  protocol hash, Git commit/tree, file hashes, interpreter, package versions,
  output directory, time limit, and behavior-affecting runtime environment
  settings. A hashed machine identity plus logical/physical CPU, process CPU
  availability, and total-memory fields prevent resume across different
  benchmark hosts or resource envelopes. Credential-bearing user information
  is removed from recorded Git remote URLs.
- A successful 444-result run writes `completion_attestation.json`, bound to
  the manifest hash, Git commit, protocol hash, expected result count, and the
  SHA-256 digest and byte size of every result pickle. It also binds and
  schema-validates `warmup_history.json` and, after any resume,
  `resume_history.json`.
- Before writing that attestation, the runner validates its own result pickles
  and writes `analysis_payload.json`, a normalized, non-executable snapshot of
  the outer metrics and child metadata. The attestation binds its digest and
  size. The standalone analyzer verifies raw-result integrity but never
  unpickles anything supplied through `--input-dir`. Before computing,
  immediately before writing a decision, and again after atomic publication,
  it rehashes the manifest, attestation, safe payload, history files, and every
  result artifact while also requiring the exact recorded source bytes and Git
  tree, clean DarkoFit and TabArena checkouts, interpreter, package versions,
  platform, hardware identity/resources, and recorded environment values.
  Analysis must therefore run from the same committed checkout, machine, and
  environment used for the campaign.
- Cache reuse requires an explicit `--resume`, and every stable manifest field
  must match. A nonempty unmanifested directory is never accepted.
- On resume, both cached arms are structurally validated against their exact
  coordinate, frozen arm, eight-child bag, fitted metadata, and refit policy.
  Validation includes AutoGluon's resolved 3,600-second outer-bag budget,
  one-set/eight-fold construction, CPU/GPU allocation, bag and fold-wise seeds,
  initial child parameters, validation/stopping metrics, and child budget
  ratios; requested hyperparameters alone are not accepted as proof.
  Before any resume mutation, every cached `results.pkl` and campaign analysis
  artifact that exists must be a real regular file; directories and symbolic
  links fail fast rather than being moved into the archive.
  If either arm is missing, unreadable, incomplete, or mismatched, every
  existing result for that coordinate is archived and both arms rerun after
  the same process-local warmup. Fully valid adjacent pairs remain cached. Any
  stale completion attestation or analysis payload is archived before work
  resumes. The five default analyzer outputs are archived at the same boundary
  so a stale report cannot remain beside newly generated results.
- `--dry-run` validates ChimeraBoost coverage, builds and orders all 444 jobs,
  and checks output-directory state without warming, writing, or fitting.

Example preflight and execution:

```bash
/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_cap_horizon.py \
  --output-dir .cache/tabarena-regression-cap-horizon-0.9.0-20260713 \
  --time-limit 3600 \
  --dry-run

/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_cap_horizon.py \
  --output-dir .cache/tabarena-regression-cap-horizon-0.9.0-20260713 \
  --time-limit 3600
```

Use the same command plus `--resume` only after an interrupted run whose
manifest is intact.

## Analysis and decision boundary

Horizon selection uses DarkoFit's paired validation results and fitted
metadata. The frozen registered `CHIMERA (default)` rows from the earlier
ChimeraBoost 0.13-era TabArena comparison are checked only for complete,
non-imputed coverage of the 222 coordinates. Their scores do not enter horizon
selection. A clean ChimeraBoost 0.14.1 checkout and CatBoost are rerun on this
machine only after a DarkoFit policy is frozen.

Report, before making a policy decision:

1. equal-dataset geometric-mean validation and test RMSE ratios, the
   hierarchical bootstrap bounds, and the t-interval sensitivity analysis;
2. per-dataset and per-repeat paired ratios, win/loss/tie counts, and the
   coordinates and ratios of their worst regressions;
3. best iteration, completed rounds, resolved learning rate, selected mode/lane,
   and stop-reason distributions for every child;
4. the fraction of children stopped by early stopping, deadline, no split, or
   iteration limit, including the fraction at or near each cap. “Near cap” is
   frozen as `rounds_completed >= ceil(0.95 * requested_horizon)`; report its
   threshold, numerator, denominator, and fraction separately for both arms;
5. paired training time, inference time, and peak-memory summaries; and
6. explicit counts of missing, failed, imputed, duplicated, or
   metadata-incomplete results, including zeros established by the analyzer's
   exact-grid and schema validation.

Per-repeat test-RMSE ratios are fold-averaged in log space and must be emitted
for every dataset/repeat block, together with fold win/loss/tie counts and that
repeat's worst split. The analyzer writes these to `per_repeat.csv` and includes
the same table in `summary.json` and `report.md` before reporting a decision.

The decision gate is frozen as follows. Ratios below one favor 10,000 rounds.
The primary estimand is paired `log(RMSE_10k / RMSE_1k)`, averaged within
repeat, then within dataset, then equally across the 13 datasets. Uncertainty
uses a pre-seeded (`20260713`) 10,000-draw hierarchical bootstrap that resamples
datasets, repeats within datasets, and folds within repeats. The advancement
gate uses the one-sided 95% upper bound (the bootstrap's 0.95 quantile); the
two-sided 95% interval is also reported. A two-sided 95% t interval over the 13
dataset estimates is reported as a sensitivity check. This wording clarifies
the statistic already implemented before the production run; it does not
change its seed, draws, quantile, or threshold.

The 10,000-round arm advances only if every requirement passes:

1. all 444 jobs and all 3,552 child metadata records are present, successful,
   finite, non-imputed, nonduplicated, and provenance-matched;
2. primary test-RMSE ratio is at most 0.995 and the hierarchical one-sided 95%
   upper confidence bound is below 1.0;
3. at least 10 of 13 dataset point estimates improve (one-sided sign-test
   `p < 0.05`);
4. no dataset has both a point ratio above 1.005 and a repeat-block bootstrap
   90% lower bound above 1.0, and no dataset point ratio exceeds 1.02;
5. equal-dataset validation-RMSE ratio is at most 1.002;
6. the mechanism is demonstrably active: at least one 1,000-round child stops
   at `iteration_limit`, and at least 20% of all 1,776 paired 10,000-round
   children execute more than 1,000 rounds; and
7. the primary panel has zero wall-clock stops, while equal-dataset training,
   inference, and peak-memory ratios are at most 2.0, 1.10, and 1.10.

Individual split regressions are reported but are not a hard gate: the earlier
`no split worse than 2%` rule was statistically unpassable at this sample size.
If the longer horizon fails any gate, retain 1,000 rounds and proceed to the
separately frozen mode, TS, representation, and linear-residual experiments.
Passing this panel freezes a candidate; it does not change defaults until the
predeclared unseen-dataset and same-machine competitor gates also pass.
