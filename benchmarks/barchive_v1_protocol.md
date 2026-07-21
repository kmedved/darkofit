# Wave 3 B-archive canonical-section feasibility protocol

_Prospective Tier-E protocol. It becomes frozen only when the create-only
contract binds this document, the runner, analyzer, freezer, tests, exact
DarkoFit source pin, the immutable M3b r3 inputs, all 13 spent case
fingerprints, and the rules below before any B-archive-v1 outcome is opened._

## Question and evidence boundary

This campaign asks one narrow question: can factoring one complete,
byte-identical numeric target-free preprocessing section plausibly bring the
M3b combined ensemble under its unchanged median archive-to-single limit of
`4.0`? It is a size-feasibility campaign, not a serializer implementation or
a revision of M3b r3.

The immutable r3 disposition remains
`close_b1_b2_preserve_existing_opt_in`. Its combined arm beat the matched
single on all 13 development cases, but failed the prospectively frozen
archive gate at `5.534767x`. The nine sports primaries are player-disjoint
cold-player rows within the held-team view. Nothing here reclassifies the
post-hoc matched-single quality readout as a frozen M3b promotion comparator.

## Fixed representative cases and models

Replay exactly the 13 spent M3b cases and their frozen data/split/weight
fingerprints: nine sports season/target cells and four medium general cells.
Use the same one-member reference and the same eight-member combined B1+B2
configuration: 600 maximum rounds, patience 30, seed 4, four threads,
without-replacement fraction 0.8, `donor_balanced_v1`, sequential fitting,
and requested shared preprocessing.

Each case runs in a fresh worker under `paired-evidence-v1`. The worker imports
the exact clean source pin in the prospectively frozen Python 3.11 / NumPy /
scikit-learn / Numba / pandas / SciPy runtime, rebuilds the frozen case, fits
the matched single and combined models, saves both through current safe NPZ,
and verifies current safe-load prediction/probability, feature-schema,
constructor/fitted metadata, and deterministic re-save identity. These checks
validate the input archives; they are not claims about a serializer that does
not yet exist.

## Only allowed simulation

The size model may remove from member archives only the complete set of
`prep__*` and `bin__*` arrays when:

- every member contains the exact same complete array-name set;
- every corresponding NPY payload is byte-identical;
- the complete preprocessing header needed to reconstruct an independent
  `FeaturePreprocessor` is identical, including the actual shared fit seed;
- fitted provenance says `shared_preprocessing="numeric_target_free"`; and
- no categorical, ordinal, target-encoder, tree, SHAP, wrapper, target, or
  other fitted member state is treated as canonical.

The simulation is deliberately non-loadable and cannot authorize a format.
For a case meeting those conditions, its effective candidate size is the
canonical-preprocessor simulated archive size. For every member-local or
otherwise ineligible case, effective candidate size is the unchanged current
ensemble archive size. Generalized member/header deltas are prohibited.

## Frozen execution and failure rules

The contract binds the existing component analyzer and exact historical
inputs. The parent verifies the source and harness trees are clean and stable,
the sports cache matches its frozen hash, and all 13 case fingerprints match
before starting workers. Results are published only as one complete,
create-only raw artifact. Any worker, invariant, or publication failure writes
a distinct create-only terminal artifact, publishes no partial rows, and makes
this identity non-rerunnable.

## Decision rule

The analyzer requires exactly 13 complete rows and all current safe-roundtrip
invariants. It also requires:

- all cases carrying `numeric_target_free` provenance to have a complete
  eligible canonical section;
- canonical array names to be only complete `prep__*`/`bin__*` names;
- no out-of-scope section to be used by the effective simulation;
- each canonical simulation to be no larger than its current ensemble; and
- member-local cases to remain byte-for-byte at current archive size.

Compute each case's effective candidate archive bytes divided by its newly
fitted matched-single bytes, then take the median across the same 13 cases.

- **Advance** only when every invariant passes and the median is `<= 4.0`.
  The disposition is `advance_to_canonical_serializer_prototype`.
- **Close** otherwise. The disposition is
  `close_barchive_nominate_fused_lane_dispatch`.

An advance funds only a new implementation prototype and a separately pinned
behavior/resource verification identity. Before any serializer can be
retained, that successor must prove exact predictions and probabilities,
feature schema, constructor/fitted metadata, safe corruption rejection,
archive round trips, and no regression in fit time, prediction time, or peak
RSS while clearing the same `<= 4.0` median archive gate. No public/default
surface, B3, M2, TabArena/M4, fresh confirmation, or lockbox access is
authorized here.
