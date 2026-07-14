# TabArena scalar-regression isolated follow-on screen result

_Executed and analyzed on 2026-07-13 from clean DarkoFit commit `704dabd7`.
The source-frozen design is in
[`tabarena_regression_followon_screen_protocol.md`](tabarena_regression_followon_screen_protocol.md)._

## Decision

**Advance only the source-declared safe ordinal representation to an
independent confirmation.** It improved equal-dataset test RMSE by 19.497%
and validation RMSE by 20.073% across Airfoil and Diamonds, won both datasets
and all six screen splits, and passed every frozen quality and resource gate.

This is an exploratory mechanism result, not a default-policy decision. The
adapter restored Airfoil's physical attack angle and the published semantic
orders of Diamonds' cut, color, and clarity fields. It never inferred an order
from labels, row order, target values, validation rows, or test rows. The
result therefore supports preserving known numeric/ordinal semantics; it does
not support treating arbitrary categoricals as ordinal.

The other four isolated mechanisms do not advance. Automatic tree-mode
selection produced a real 3.099% RMSE improvement but a 2.574x inference-time
ratio, above the frozen 1.25x ceiling. Four target-statistic permutations
failed the aggregate, majority, and dataset-harm gates. Safe one-hot failed
the strict-majority gate, and linear-residual boosting failed the dataset-harm
gate.

## Primary screen

Ratios below one favor the candidate. Each candidate changed exactly one
lever from the shared 1,000-round CatBoost/L2=3/128-bin/LR=0.1/TS1 control.

| Arm | Scope | Test RMSE | Validation RMSE | Train time | Infer time | Peak memory | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `auto` | 13 datasets | 0.969013 (-3.099%) | 0.964812 (-3.519%) | 3.627482 (+262.7%) | 2.574409 (+157.4%) | 1.016780 (+1.7%) | reject: inference |
| `ts4` | 5 datasets | 0.997888 (-0.211%) | 0.998632 (-0.137%) | 0.957052 (-4.3%) | 0.978984 (-2.1%) | 1.007591 (+0.8%) | reject: quality/harm |
| `ordinal` | 2 datasets | **0.805027 (-19.497%)** | **0.799266 (-20.073%)** | 1.191638 (+19.2%) | 1.222666 (+22.3%) | 1.027140 (+2.7%) | **advance** |
| `onehot` | 6 datasets | 0.960005 (-4.000%) | 0.953191 (-4.681%) | 1.143714 (+14.4%) | 1.163125 (+16.3%) | 1.004668 (+0.5%) | reject: majority |
| `linear` | 13 datasets | 0.989259 (-1.074%) | 0.988244 (-1.176%) | 0.961184 (-3.9%) | 1.049913 (+5.0%) | 1.027994 (+2.8%) | reject: harm |

All ratios are paired log ratios averaged within dataset and then equally
across datasets. The analyzer used outer test RMSE for this research screen;
the three coordinates `r0f0`, `r1f1`, and `r2f2` are spent.

## Ordinal survivor

| Dataset | Test RMSE change | Validation RMSE change | Split wins |
| --- | ---: | ---: | ---: |
| Airfoil self noise | -15.729% | -13.335% | 3/3 |
| Diamonds | -23.097% | -26.287% | 3/3 |

- Hierarchical 95% test-RMSE ratio interval: **[0.755375, 0.850959]**.
- One-sided 95% upper bound: **0.848938**.
- Dataset wins/losses/ties: **2/0/0**.
- Split wins/losses/ties: **6/0/0**.
- All 48 child fits used the source-frozen target-free representation with
  zero validation-time unknown values.
- Stop reasons were 36 early stops and 12 iteration-limit stops; no child hit
  the wall-clock deadline.

Strict TabArena v0.1 contains no regression dataset untouched by the preceding
cap campaign. The confirmation must therefore either use independently frozen
external datasets with declared ordinal semantics or, as a weaker
mechanism-level holdout, the remaining registered Airfoil and Diamonds
coordinates. Any product implementation must remain explicit and fail closed;
automatic lexical, frequency, or target-derived ordering is out of scope.

## Rejected mechanisms

### Automatic tree mode

`tree_mode="auto"` improved 11 of 13 dataset point estimates and selected 133
CatBoost, 97 hybrid, and 82 LightGBM children. Its gain was concentrated in
Diamonds (-22.684%), Airfoil (-5.106%), and Physiochemical Protein (-4.792%);
the other ten datasets improved by about 0.5% in aggregate. The inference
penalty is genuine: the fitted ensemble retains only the selected child model,
not all three candidates, and every dataset exceeded the 1.25x inference
ceiling. It remains an explicit accuracy-oriented option, not a default-policy
survivor.

### Four target-statistic permutations

`ts_permutations=4` improved only two of five applicable datasets, missed the
0.5% aggregate improvement threshold, and regressed Diamonds by 0.856%. It
does not advance.

### Safe one-hot

The 4.000% aggregate gain was almost entirely Diamonds (-21.657%). Excluding
Diamonds, the other five datasets improved by only 0.017% in aggregate;
Healthcare regressed 0.252%, while Miami and Wine were exact no-ops at the
child boundary. The transform was target-free and correctly left Food
Delivery's high-cardinality identifier on the native categorical path, but it
did not show broad policy value.

### Linear residual

Linear residual improved seven datasets and overall RMSE by 1.074%, but
QSAR-TID-11 regressed 1.783%, violating the 0.5% dataset-harm ceiling. Its
aggregate gain was also dominated by Diamonds (-12.945%); excluding Diamonds,
the remaining 12 datasets were effectively neutral.

Diamonds is a shared native-categorical weakness across several mechanisms,
not four independent broad-policy confirmations. It should be investigated
separately from any generic default change.

## Integrity

- **156/156** outer jobs succeeded: 39 shared controls and 117 isolated
  candidate comparisons.
- **1,248/1,248** child-fit records were complete and unique.
- All 936 automatic-mode candidate fits completed; no candidate or selected
  model hit its deadline.
- There were zero failures, imputations, cache substitutions, missing rows,
  duplicates, or wall-clock stops.
- Every one of the 156 raw-result byte hashes and sizes matched the completion
  attestation. The manifest, safe payload, warmup record, dependency lock,
  hardware identity, Git trees, and all 16 bound source files also matched.
- A second analyzer execution produced byte-identical CSV, JSON, and Markdown
  outputs. An independent recomputation matched all 2,925 payload-to-table
  numeric fields and the seeded 10,000-draw bootstraps.

## Retained evidence

The repository retains the analyzer's machine-readable
[`summary`](tabarena_regression_followon_screen_summary.json),
[`paired splits`](tabarena_regression_followon_screen_paired_splits.csv),
[`per-repeat estimates`](tabarena_regression_followon_screen_per_repeat.csv),
[`paired child metadata`](tabarena_regression_followon_screen_paired_children.csv),
[`run manifest`](tabarena_regression_followon_screen_run_manifest.json),
[`completion attestation`](tabarena_regression_followon_screen_completion_attestation.json),
and [`warmup record`](tabarena_regression_followon_screen_warmup_history.json).
The 6.3 MB safe analysis payload and 53.4 MB of raw result pickles remain in
the hash-addressed local campaign directory; the committed attestation binds
their hashes and sizes.

## Provenance

- DarkoFit commit: `704dabd7ac35454678544331f7c803182bead128`;
  Git tree: `2bba6b6acb9fedff90d7844520bf21de2e19c1b0`.
- Python: 3.12.13; AutoGluon: 1.5.1b20260712; TabArena commit:
  `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.
- Run manifest SHA-256:
  `3299884d5315b25c0e6437e82c2bfaf638a16f765a4397838586988c0c009498`.
- Completion attestation SHA-256:
  `1cd255aeed7bf5ec0c2f3cfe0d553584730fdef2da472a22aa247832e21fc440`.
- Safe analysis payload SHA-256:
  `9eee6f558af17b2215d408751b0e931262c9df81f91bf1cd5eda251cad56a447`.
- Analyzer summary SHA-256:
  `9ad876db5775df324df3efad23620c22edd964406c706186b317ea802dce7bcd`.
- Paired-split table SHA-256:
  `de4bdc55bb332f0eb4c41d52191909e0ef43d185fabac80d8f259e1b8758b694`.
- Paired-child table SHA-256:
  `56effbe84091ed7f6ad5c939579468eeafc85806f5fde60335876c4585e724eb`.
- Frozen protocol semantic digest:
  `85c166afec3aed94c7b9e5d509f9e4186c8ebe0361b2e1ed39fc442f1a0762c4`.

The standalone analyzer never unpickled result files. It revalidated the
complete attested campaign immediately before and after atomically publishing
the decision artifacts.
