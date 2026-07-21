# Wave 2 M3b private ensemble-v3 attribution protocol, attempt 3

_Prospective successor amendment. It becomes frozen only when the create-only
attempt-3 contract binds both prior protocols/contracts/failure records, the
corrected implementation and tests, successor harness, new source pin, exact
case fingerprints, and unchanged decision rules._

## Why attempt 2 is terminal

Attempt 2 passed RSS, source, and data preflight. Its first single-reference
row completed in memory; the next group-bootstrap control then failed during
the mandatory safe-NPZ reload. The create-only terminal artifact discarded
the completed row and published no model result. No value from that row was
inspected. Its digest and disposition are preserved in
`m3b_ensemble_v3_attempt2_failure_record.json`; attempt 2 must not be rerun.

The failure exposed a private serialization validator bug. Row bootstrap
draws exactly `input_rows` rows, but group bootstrap draws exactly the input
number of groups. Unequal group sizes mean the resulting sampled row count is
not generally `input_rows`. Fitting and saved metadata correctly represented
that plan; safe loading incorrectly applied the row-bootstrap size invariant
to both sampling units.

## Sole model-source correction

Attempt 3 pins the commit that limits `sampled_rows == input_rows` to row
bootstrap. Group bootstrap continues to require all stronger applicable
invariants: positive counts, sampled-unique plus OOB row complement, exact
group-draw/unique/OOB relation, group-disjointness, member policy provenance,
and fitted-member consistency. A deterministic uneven-group regression test
now proves save/load prediction and metadata identity. The M3b analyzer's
matching evidence check accepts variable sampled row counts only for group
bootstrap.

This correction changes persistence acceptance for valid private group
bootstrap models; it does not change sampling, fitting, predictions, data,
quality metrics, arm order, or decision thresholds. Attempt 3 therefore uses
a new model source pin and contract identity.

## Everything else remains fixed

Attempt 3 retains attempt 2's self-worker RSS capability preflight and scope.
All arms, 13 cases/splits/weights, four-thread worker environment, warmup,
fit settings, OOB telemetry, quality-first gates, selective timing repeats,
resource/value thresholds, terminal-failure rules, and non-shipping claim
boundaries remain exactly as prospectively declared by the attempt-1 and
attempt-2 protocols. Neither prior attempt supplies a model outcome to this
campaign.
