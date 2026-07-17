# Basketball categorical-combinations donor-screen result

_Formal result for
[`basketball_categorical_combinations_protocol.md`](basketball_categorical_combinations_protocol.md)._

## Decision

**Close ChimeraBoost's pairwise categorical-combinations mechanism without a
DarkoFit port.**

The candidate failed every preregistered quality gate. It also made public
prediction materially slower. No DarkoFit implementation, explicit parameter,
automatic policy, default change, broader dataset campaign, or lockbox spend
is authorized.

The immutable artifact is
[`basketball_categorical_combinations.json`](basketball_categorical_combinations.json).

## Quality

The frozen all-categorical basketball view used `Pos`, exact categorical
`Age`, `Tm`, and `starter`; `Player` was excluded as a model feature and used
only for grouped internal validation and the cold-player boundary. The
candidate added all six pairwise categorical combinations.

| View | Combinations off R² | Combinations on R² | Candidate delta |
| --- | ---: | ---: | ---: |
| Mean of ten creator folds | **0.561681** | 0.560656 | **-0.001024** |
| Overlap-exposed held teams | **0.560753** | 0.514265 | **-0.046489** |
| Genuinely cold players | **0.537427** | 0.446023 | **-0.091404** |
| Seen-player subset | **0.558020** | 0.524775 | **-0.033245** |

The candidate won five folds and lost five; the gate required at least six
wins. Its worst fold delta was `-0.012930`, beyond the `-0.010` floor. The
mean-fold, held-team, cold-player, and seen-player floors all failed.

This is not a marginal sports failure. On the primary unseen-player guardrail,
the pairwise columns erased about 0.0914 R². Team-bearing combinations cannot
generalize to the alphabetically held-out teams, and the remaining
position/age/starter combinations did not compensate.

## Runtime and behavior

| Metric | Candidate/control median ratio | Gate | Result |
| --- | ---: | ---: | --- |
| Fit | 1.115x | ≤1.50x | pass |
| Held-team prediction | 2.843x | ≤1.10x | fail |
| Cold-player prediction | 1.653x | ≤1.10x | fail |
| Peak RSS | 1.002x | ≤1.50x and ≤256 MiB added | pass |

Every timing series was stable under the frozen IQR/median limit. The behavior
checks also passed:

- every candidate fit created exactly the six declared pair columns;
- every control fit created none;
- all workers resolved 18 threads and learning rate 0.1;
- repeated full fits produced array-exact predictions and identical fitted
  structure within each arm;
- on the unchanged numeric basketball view,
  `cat_combinations=None` and `False` produced exact predictions, feature
  importance, and tree counts, with no combination route engaged.

These checks show that the rejection is about the intended donor mechanism,
not a route failure or a noisy run.

## Scope and provenance

- Formal source: clean `main` at
  `713706638e6e52127866cf78e9a3b3a19b4b5934`, equal to `origin/main`.
- ChimeraBoost: clean local, origin, and upstream `main` at
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`0.15.0`).
- Runtime: Python 3.12.13 on Apple M5 Max with 18 logical CPUs and the frozen
  dependency stack.
- Independent pre-run review found no remaining issues; the full suite passed
  with 1,622 tests and 23 skips.
- Artifact size: 2,038,643 bytes.
- Artifact SHA-256:
  `c4cede0223af036ccf3d2e4f207eaa6fd7e272a87faa24ee6fcb2361ba09379c`.
- No ChimeraBoost source was copied, and no CTR23, TabArena, or lockbox
  evidence was used.

Basketball remains the fast fatal screen. A future categorical proposal must
be a materially different mechanism with a new frozen protocol; retuning this
pairwise-combination candidate is closed.
