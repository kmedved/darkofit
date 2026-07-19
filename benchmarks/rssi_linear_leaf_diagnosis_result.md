# RSSI linear-leaf parity diagnosis: result

**Outcome:** the apparent RSSI linear-leaf implementation gap does not exist.
With identical training rows, validation rows, and parameters, DarkoFit and
ChimeraBoost 0.15 are behavior-exact in both constant- and linear-leaf lanes.
The product gap on this coordinate is a defaults and selection-policy gap.

This is spent-data mechanism evidence from one OpenML fold. It cannot promote
a default, estimate broad generalization, or support a timing claim.

## Evidence bindings

- Frozen raw file SHA-256:
  `02c2d36a12b3a452363cc9b8a62b1cf246b09829a5544d233eae375b10d17ef6`
- Original source commit:
  `dcd6e298e61aaf114d922cef4e1666fefcd66add`
- Original run-time runner SHA-256:
  `136296297733f24d31f5bc82ad049411f1baec806a88497821c38ff1e4771c05`
- Current hardened source commit:
  `816101476bb65cf5a0e2f59cd11edaf96f46a1cc`
- Current hardened runner/verifier SHA-256:
  `8b4b9ec41cfa9178ff93c143bd9d09abce98cf0194f943ed9c003a145e944104`

The original hash identifies the runner bytes that produced the frozen raw
artifact. The current hash identifies the later hardened copy used to
revalidate that artifact and its embedded analysis; it did not generate new
benchmark outcomes.

## Reproduction

```bash
/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_rssi_linear_leaf_diagnosis.py
```

- DarkoFit source: `dcd6e298e61aaf114d922cef4e1666fefcd66add`
- ChimeraBoost source:
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`v0.15.0`)
- Protocol SHA-256:
  `9a0465dfb0273bf5289cc6a91f160462c08c1e075cdfa972ba2bb6aadeafbcaf`
- Data: OpenML task `363132`, repeat 0, fold 0, sample 0
- Shared inner validation: 20% of the official training fold,
  `ShuffleSplit(random_state=4)`
- Full artifact:
  [`rssi_linear_leaf_diagnosis.json`](rssi_linear_leaf_diagnosis.json)

## Results

| Arm | Test RMSE | Best validation RMSE | Trees | Selected |
|---|---:|---:|---:|---|
| DarkoFit default | 0.000700604 | — | 1,000 | current default |
| DarkoFit matched, automatic 10% validation | 0.000090636 | 0.000085993 | 759 | forced linear |
| DarkoFit matched, automatic 20% validation | 0.000079643 | 0.000081676 | 728 | forced linear |
| DarkoFit shared constant | **0.000066109** | **0.000054399** | 607 | forced constant |
| DarkoFit shared linear | 0.000079643 | 0.000081676 | 728 | forced linear |
| ChimeraBoost shared constant | **0.000066109** | **0.000054399** | 607 | forced constant |
| ChimeraBoost shared linear | 0.000079643 | 0.000081676 | 728 | forced linear |
| ChimeraBoost full selector | **0.000066109** | **0.000054399** | 607 | constant |
| ChimeraBoost 100-round selector | 0.000079643 | 0.000081676 | 728 | linear |
| ChimeraBoost full product selection | **0.000066109** | **0.000054399** | 607 | constant; no cross |
| ChimeraBoost product default | 0.000079643 | 0.000081676 | 728 | linear; no cross |

Fit seconds are present in the JSON only for auditability. The run did not use
a timing protocol, so they are intentionally omitted here.

## What is exact

For each matched DarkoFit/ChimeraBoost pair, all of the following are
identical:

- fitted borders;
- complete validation history;
- normalized full-ensemble tree fingerprint;
- prediction bytes;
- retained tree count;
- best validation RMSE; and
- outer test RMSE.

DarkoFit's automatic 20% linear run is also exact to its explicit shared-split
linear run. This rules out the binner, tree builder, linear solver, prediction
path, and explicit-versus-automatic split application as explanations for the
matched-lane result.

## Diagnosis

1. **The historical 8.8× DarkoFit-default gap is policy, not missing
   machinery.** The matched full-budget constant lane is 0.8301× the
   ChimeraBoost product RMSE on this coordinate.
2. **Validation size matters materially here.** Holding the model family and
   parameters fixed, DarkoFit's 10% validation policy produces 1.1380× the
   RMSE of the exact same 20% policy.
3. **The 100-round linear-leaf audition picks the wrong eventual winner.**
   Full-budget validation selects constant leaves. The capped audition selects
   linear leaves, whose test RMSE is 1.2047× the constant lane.
4. **Cross features do not explain this task.** ChimeraBoost declines them
   under both full and capped product selection.

The result does not prove that 20% validation or constant leaves should become
global defaults. It proves that RSSI is not evidence for porting another
linear-leaf implementation or crediting cross features for the observed gap.

## Consequences for T5/T6

- Preserve DarkoFit's current behavior-exact linear-leaf engine.
- Do not copy ChimeraBoost's 100-round constant-versus-linear selector into
  the T5 composite. A frozen candidate must compare lanes at a budget shown to
  preserve their full-race ordering, or use a separate guard whose regression
  profile is evaluated as the candidate.
- Continue diff/product feature development on other spent smooth/geometry
  tasks. RSSI remains useful as a validation/default-policy case, not as the
  motivating cross-feature case.
- Include validation-fraction policy in T5's frozen ablation and cost model.
