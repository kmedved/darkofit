# Fresh confirmation registry v2 amendment

## Correction

Registry v1 remains immutable and valid for task identity, contamination,
lineage, coordinates, and power. Its primary stratum label,
`smooth_numeric`, is too narrow.

The pre-score audit compared the downloaded semantic fingerprints with the
OpenML task-list metadata used during candidate curation. OpenML reported zero
symbolic features for several tasks whose downloaded dataframe contains
categorical dtypes. Of the 14 primary mechanism tasks:

- 5 are numeric-only and complete;
- 7 contain categorical features but no missing predictors; and
- 2 contain categorical features and missing predictors.

No model, target statistic, or candidate result was read. The error affects
the descriptive stratum label, not the target-blind selection, lineage
allocation, coordinates, semantic fingerprints, or power calculation.

## Binding amendment

The v2 artifact:

1. binds the exact v1 file and canonical registry hashes;
2. retains all 20 task IDs, lineages, eligibility decisions and 60
   coordinates;
3. renames the 14-task primary stratum to `smooth_process`;
4. records actual categorical/missingness profiles from v1 fingerprints; and
5. retains the 3 categorical and 3 noisy-tabular guardrails.

Future runners must bind v2 and describe the primary claim as smooth/process
mechanism confirmation, not a numeric-only smooth panel. V2 does not score
data, promote the selector, or authorize the CTR23 lockbox.
