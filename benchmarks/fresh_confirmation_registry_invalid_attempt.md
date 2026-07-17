# Invalid fresh-registry attempt

The first clean invocation downloaded and fingerprinted the declared tasks but
failed closed before writing an artifact. The repository exposure check used
`git grep -F` on bare numeric OpenML task IDs. Historical JSON benchmark
artifacts contain arbitrary floating-point decimal strings, so every six-digit
task ID appeared incidentally inside unrelated measurements.

This was not contamination evidence. Exact OpenML task and dataset IDs remain
structured fields in the registry and exposure catalogs; task names remain
source-grepped; normalized-name, exact dataset-ID, exact semantic fingerprint,
and target-blind near-lineage checks remain binding.

No model was fit, no target statistic or benchmark result for a candidate task
was read, no registry artifact was written, and no candidate, coordinate,
threshold, or power rule changed. The complete target-blind build must be
rerun from clean committed source.
