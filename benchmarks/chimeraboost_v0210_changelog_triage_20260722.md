# ChimeraBoost 0.21.0 changelog triage

_Recorded before the catcross candidate was implemented or any candidate
outcome was inspected._

## Source pin

- Release tag: `v0.21.0`, commit
  `26fed8a715fe172518472f4fec1a663492db6f61` (2026-07-22).
- Group-centered-cross donor commit:
  `e0d401bab9b16041f0323ae923e43dfa413532c3`, an ancestor of the release
  tag. The mechanism shipped in 0.20.0 and remains in 0.21.0.
- License: Apache-2.0. DarkoFit's existing NOTICE-based donor practice
  applies if source is adapted.
- Release-file SHA-256 values: `CHANGELOG.md`
  `ba79c81d8a71d415461f987896b250b421aabec7edb5249cd53c5dc470d77fb6`;
  `chimeraboost/preprocessing.py`
  `99cb13ba51a56bd59c1556735b8a768f2f005d5598455af9488a9c68183180fe`;
  `chimeraboost/sklearn_api.py`
  `d34352c6ab6367e717cb93b3fa4a0191f5ba51ae94e1e46feb7966978add51c0`;
  `tests/test_gdiff_crosses.py`
  `65246d0dc516883bdde5f77bc38f76b3f653a106c450e34b1cc6764750e22a9e`;
  `LICENSE`
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

The release tag, not a moving checkout, is the evidentiary source. Local
branch or origin drift cannot silently change this intake.

## 0.21.0 changes

| Change | DarkoFit disposition | Reason / next gate |
| --- | --- | --- |
| Remove pandas as an install dependency and use native categorical transforms | No campaign | DarkoFit already has no runtime pandas dependency and uses pandas only as an optional loaded-module fast path. No product gap is established. |
| Share categorical transform work across bagged members at predict | Track I, prediction/memory shortlist; sequence with B3 | Directly relevant to ensemble-v3's measured prediction cost. First characterize B3's fixed-topology member execution and the existing shared-preprocessing surface; any port must preserve member-specific target-stat maps and exact predictions. |
| Restore the parent thread budget to parallel-fitted members before sequential prediction | Audit inside B3 before copying | Same user-visible symptom class as DarkoFit's earlier thread-state work. B3 must test fitted member masks and aggregate prediction CPU use under a fixed total budget. A confirmed DarkoFit bug is fixed as hygiene; otherwise no donor project. |

## 0.20.0 mechanism carried into the pin

**Fund `group_centered_categorical_crosses_v1` as the current quality slot.**
For a numeric feature `x` and categorical feature `c`, add the target-free
column `x - mean_fit(x | c)`, using fit rows and weights only and falling back
to the global fit mean for unseen categories. This turns “above the baseline
for this row's category” into one split.

This is materially different from the closed pairwise categorical-
combinations donor: it combines a numeric feature with a categorical grouping
through a target-free centered value; it does not create category-by-category
keys or target-encode their Cartesian product. The old sports failure does not
pre-adjudicate this mechanism.

Expected value is high but localized: the v0.11 ladder's largest quality miss
was categorical (`diamonds` at `1.386479×` versus ChimeraBoost 0.20), while
DarkoFit led the other twelve cases by about 1.2% after removing diamonds.
The donor reports small broad average gains and a flat independent gate, so
the candidate must be automatic and exact when it declines; it is not assumed
to improve all categorical data. The first implementation is bounded to
single-model scalar-RMSE CatBoost mode and at most 12 added columns (top four
numeric by base-fit importance crossed with top three categorical features).

Falsifiable development stop rules are the immutable M6 v3 rule: aggregate
primary-loss ratio `<=1.000`, worst dataset ratio `<=1.020`, and worst
leave-one-dataset-out ratio `<=1.003`. Failure kills this exact candidate.
Advancement authorizes only mechanism-specific spent attribution, not a merge,
default, API, fresh panel, release, or quality claim.
