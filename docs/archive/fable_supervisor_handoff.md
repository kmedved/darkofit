# Supervisor Handoff — ChimeraBoost distributional work

**Written:** 2026-07-08 by the outgoing supervisor agent (Claude Fable 5).
**For:** the next agent taking over supervision on a fresh machine.
**One-line state:** distributional regression is shipped (v0.6.0, on `main`, 513 tests green); the Kalman-readiness investigation reached *statistical parity* with the incumbent and the final call now belongs to a replay inside the external DARKO filter; a new feature thread (linear residual boosting) exists as a plan only.

---

## 0. Fast facts

| | |
|---|---|
| Repo URL | `https://github.com/kmedved/chimeraboost.git` (public) |
| Default/working branch | `main` (tracks `origin/main`, in sync) |
| Latest commit | `18dad32 Merge remote main` |
| Release | `v0.6.0` tagged + GitHub release with built wheels; **not on PyPI** (blocked on credentials) |
| Package version | `0.6.0` (dynamic from `chimeraboost.__version__`) |
| Python | requires `>=3.9`; **CI uses 3.11** — match it |
| Runtime deps | `numpy>=1.22`, `numba>=0.57`, `scikit-learn>=1.0` |
| Extras | `dev = [pytest, optuna>=4, pandas]`, `tuning = [optuna>=4]` |
| Test suite | **513 passed** (verified this session) |
| Git owner | `kmedved` (git user); this is a personal repo |

**Dirty git status right now** (both are untracked, neither committed):
- `?? LINEAR_RESIDUAL_BOOSTING_PLAN.md` — next-feature plan (see §7).
- `?? fable_supervisor_handoff.md` — this file.

Everything else (distributional feature + all replay work) is committed and merged.

## 1. Project path & what is machine-specific

Path on the machine this was written on: `/Users/kmedved/Code/GitHub/chimeraboost`.

| Portable (travels via git / clone) | Machine-specific (will NOT exist on a new machine) |
|---|---|
| The whole repo (clone from URL) | The repo path above (clone anywhere) |
| All `*.md` specs, `chimeraboost/`, `tests/`, `benchmarks/bench_distributional.py` (synthetic) | **WNBA data:** `/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq` (private, local) |
| Full test suite | The `wnba_darko` repo (separate Dropbox checkout) — the production Kalman gate lives there |
| | Owner memory dir: `/Users/kmedved/.claude/projects/-Users-kmedved-Code-GitHub-chimeraboost/memory/` |
| | Session scratchpad dir (regenerated per session) |

**Consequence:** on a new machine you can do *everything library-side* (specs, code review, synthetic benchmark, the whole test suite). You **cannot** run any `bench_wnba_*` script or the Kalman replay — they need the private parquet and fail with a clear message without it. That is expected, not a bug.

## 2. Bootstrap a new machine

```bash
git clone https://github.com/kmedved/chimeraboost.git
cd chimeraboost
python3.11 -m venv .venv && source .venv/bin/activate     # 3.9+ works; CI is 3.11
pip install -e ".[dev,tuning]"                            # pytest, optuna, pandas
python -m pytest -q                                       # expect: 513 passed (cold ~40–60s: numba JIT warms on first run)
python -c "from chimeraboost.losses import VECTOR_LOSSES; print(list(VECTOR_LOSSES))"
#   -> ['Gaussian', 'LogNormal', 'StudentT', 'Poisson', 'NegativeBinomial']
```

Note: on the origin machine the package was used *from the tree* (not pip-installed), so throwaway verification scripts needed `PYTHONPATH=/path/to/chimeraboost`. On a new machine, `pip install -e` above removes that need. Put throwaway scripts in the session scratchpad dir, never `/tmp`.

## 3. Commands already run this session, and outcomes

| Command | Outcome |
|---|---|
| `python -m pytest -q` (re-run each review) | **513 passed** (grew 330→451→…→513 across the stream) |
| `git status / log / gh pr view 6 / gh run list` | on `main`, synced; PR #6 MERGED (`5412ebc`); CI green on `main` |
| Independent FD gradient checks on all 4 new heads (custom script) | worst rel err ~1e-10 — kernel math correct |
| LogNormal ⟷ Gaussian-on-log-y raw-score equivalence (custom) | **bit-identical** (max diff 0.0) |
| StudentT `variance_from_raw` vs `scale²·ν/(ν−2)` (custom) | exact |
| Poisson λ recovery / non-integer rejection (custom) | corr 0.981; rejects correctly |
| NB overdispersion (custom) | r̂≈2.4 vs true 2; Var>mean |
| 200k×20 perf/quality smoke (custom) | Gaussian fit 1.60× RMSE fit; σ̂-corr 0.994; 90% coverage 0.903 |
| Per-metric NLL decomposition of Kalman round-1 (custom) | `pace` = 91% of the gap → the "loss" was a bad-floor artifact, not real |

All custom verification scripts were throwaways in the scratchpad; recreate as needed (patterns in §8).

## 4. Task state: done / partial / not started

**DONE (shipped in v0.6.0, on `main`, verified):**
- Gaussian distributional head (μ + log σ, natural gradient) + full wrapper surface: `predict` (mean), `predict_dist`, `predict_variance`, `predict_interval(alpha)`, `sample`, `staged_predict`.
- Calibration `dist_calibration=` (`"scalar"`/`"affine"`/`"per_metric_affine"`; old `sigma_calibration=` is a 1-release deprecation alias). Fit on a validation fold, frozen through refit, persisted through save/load.
- General distribution protocol + 4 more heads: `LogNormal`, `StudentT` (fixed-ν), `Poisson`, `NegativeBinomial` (global-dispersion). (M0–M4 of `DISTRIBUTIONAL_HEADS_SPEC.md`.)
- Tuner, serialization, auto-LR all generalized to the heads. W0 metric-consistency fix (unclipped CRPS, `|z|≤1000` NLL overflow guard, training keeps `|z|≤10` clip).
- CI (`.github/workflows/tests.yml`, py3.11, numba-kernel cache). GitHub release + wheels.

**PARTIAL / IN FLIGHT:**
- **Kalman readiness** — library side done; the *use* of σ̂² as filter `R_t` reached statistical parity on a toy replay (§6). Real gate (production DARKO filter) not run.
- **Per-metric G1 calibration** — pooled bins ~pass; `pf_100`/`pts_100` slices still miss strict G1.

**NOT STARTED:**
- **PyPI publish** (credentials — owner-only).
- **NB heterodispersion (M5)** — designed & gated in `DISTRIBUTIONAL_HEADS_SPEC.md` §4.2; build only if real count data needs it.
- **Linear residual boosting** — plan only, untracked (§7).

## 5. Files: changed / important, and what each does

Source (`chimeraboost/`):
- `losses.py` — all loss classes incl. the 5 `VECTOR_LOSSES` heads + their numba kernels (grad/hess, eval, init). **First stop for any head math.**
- `booster.py` — `DistributionalBoosting` (fit loop, `predict_raw`/`predict_dist`/`predict_variance`); scalar + multiclass boosters. Per-head ρ-LR / ρ-L2 knobs live here.
- `sklearn_api.py` — `ChimeraBoostRegressor`/`Classifier`; calibration fitting/persistence; the public `predict_*` methods.
- `tree.py` — shared vector-tree builder (K-generic; heads ride it unchanged) + per-class L2 plumbing.
- `serialization.py` — npz save/load; `loss_state`; `n_outputs`-vs-loss consistency check.
- `tuning/{search,scoring,spaces}.py` — Optuna tuner, generalized to the heads.

Planning / spec docs (repo root — the format precedent for any new spec):
- `KALMAN_READINESS_PLAN.md` — workstreams W0–W7, gates G1–G5, the W0 adjudication, the Kalman verdict. **Start here for the Kalman thread.**
- `DISTRIBUTIONAL_HEADS_SPEC.md` — head protocol + StudentT/Poisson/NB/LogNormal specs (M0–M4 done, M5 gated).
- `DISTRIBUTIONAL_REGRESSION_SPEC.md` — original Gaussian head spec. `DISTRIBUTIONAL_REVIEW.md` — its review.
- `LINEAR_RESIDUAL_BOOSTING_PLAN.md` — next feature, plan only (untracked).
- `ROADMAP.md`, `BENCHMARK_NOTES.md`, `CHANGELOG.md`, `README.md`.

Benchmarks (`benchmarks/`):
- `bench_distributional.py` — synthetic promotion benchmark. **Runs anywhere.**
- `bench_wnba_realdata_distributional.py` — real-data calibration check. **Needs the parquet.**
- `bench_wnba_kalman_replay.py` + `wnba_kalman_replay_summary.md` — the toy replay. **Needs the parquet; saturated — do not keep tuning it (§6).**

Tests: `tests/test_distributional.py` (heads/calibration), plus `test_chimeraboost.py`, `test_tuning.py`.

## 6. The Kalman verdict (read carefully — easy to misread)

Goal: use distributional σ̂² as the DARKO Kalman **observation noise `R_t`**. Binding gate (`KALMAN_READINESS_PLAN.md` G1): per-slice standardized-residual RMS ≈ 1 (a filter consumes only second moments).

- Real-data one-step calibration check **passed** strongly (NLL 1.435 → 0.423 vs constant-σ).
- Toy scalar replay round 1 *looked* like a loss; decomposition showed **91% of the gap was one metric (`pace`) pinned at a global `r_floor` 5× too large**. Artifact, not a real loss.
- Round 2 (per-metric floors, StudentT(ν=30) + validation-tuned incumbent blend): **statistical parity** — overall NLL gap `2.6e-5` nats (a tie; the summary itself prints `-0.0000`), better innovation calibration (NIS 0.994 vs 0.982). Did **not** clear the strict 2-of-3-season gate.

**Honest conclusion (Codex's "clean win" framing overstated it):** the toy harness is **saturated**. The incumbent `σ²/sample_weight` heuristic is already near-optimal for 1-D filtering; the tuner even chose *pure incumbent* for 3 of 6 metrics. **Stop tuning the toy replay** — a zero-nat gap after ~10 tried lane-families is optimizing noise / selecting on test. What's established: parity on likelihood, better second-moment calibration, per-metric floors are mandatory, and the blend is the right production contract. The decision moves to the real multi-metric player filter.

## 7. Next feature thread: linear residual boosting

`LINEAR_RESIDUAL_BOOSTING_PLAN.md` (untracked) is a **plan only, nothing implemented** — port the `lrboost` idea (linear model, then boost trees on residuals) as **opt-in wrapper functionality on `ChimeraBoostRegressor`** (not a core objective, not a new estimator). Already through two Oracle rounds. Unrelated to the Kalman thread. If picked up, run the standard loop: plan → Codex-executable spec → adjudicate reviews → verify.

## 8. Open questions, blockers, risks, assumptions

**Blockers:**
- PyPI publish needs credentials (owner-only).
- Production DARKO replay needs the external `wnba_darko` repo + private data — not runnable on a generic new machine.

**Risks / assumptions to hold onto:**
- **Codex overclaims.** It reported a "real, fair win" on a `2.6e-5`-nat tie selected best-of-ten. Treat tiny-gap / best-of-N / aggregate-over-slices results as suspect; demand noise bands.
- **Uncommitted-work risk is recurring.** Multiple times this stream, hundreds–thousands of lines sat uncommitted. Check `git status` every review; tell the owner to commit; keep unrelated changes in separate commits.
- **Selection-on-test** crept into the Kalman lane choice (family picked by test-season performance). The production gate must use paired-bootstrap noise bands so ties read as ties.
- **Assumption:** heads are K-generic on the vector-tree path (verified — no `K>=2` assumptions in `tree.py`). Any new head that violates the positive-Hessian / zero-weight-skip invariants (`DISTRIBUTIONAL_HEADS_SPEC.md` standing invariants) will break split legality.

**Open questions:**
- Does the learned `R_t` beat the incumbent in the *real* player filter (not the saturated toy)? Unresolved — this is the whole remaining Kalman question.
- Are the `pf_100`/`pts_100` per-metric G1 misses worth chasing (per-metric StudentT / ρ-head reg), or accept them?

## 9. Exact next steps (pick with the owner)

1. **Commit the two untracked docs** if they should travel (this handoff + the lrboost plan), or hand them over out-of-band. On a *new machine* they only arrive via git.
2. **Owner: publish v0.6.0 to PyPI** (add API token / trusted publishing, then `twine upload` the already-built wheels).
3. **Production DARKO replay** (task chip `task_8057f8f7`, runs in `wnba_darko`): inject the *blended* `R_t` (validation-tuned per-metric mix of ChimeraBoost `predict_variance()` and incumbent), per-metric floors, StudentT(ν=30) + `per_metric_affine`, and **report paired-bootstrap noise bands**. Adopt only on a clear 2-of-3-season win outside the band; else document parity and keep incumbent with the blend behind a flag. (Likely a well-evidenced "keep incumbent" close — that's a valid outcome.)
4. **If the owner pivots to linear residual boosting:** turn `LINEAR_RESIDUAL_BOOSTING_PLAN.md` into a Codex-executable spec (use `DISTRIBUTIONAL_HEADS_SPEC.md` as the format), then implement → verify.
5. **Optional polish for a v0.6.1:** per-metric G1 edges (`pf_100`/`pts_100`); NB M5 only if real count data demands it.

## 10. Copy-paste prompt to start the next worker

```
You are taking over supervision of the chimeraboost project (github.com/kmedved/chimeraboost,
branch main). Read fable_supervisor_handoff.md in the repo root first — it has full state.

Your role is architect/adjudicator/independent-verifier over a "Codex" implementer worker and
pasted-in "Oracle" reviews — NOT primary coding. Your leverage is judgment and verification.

Ground rules, in order:
1. Never accept a "done / N passed" report at face value. Re-run `python -m pytest -q` yourself
   (expect 513 passed), read the actual diff, and for anything numeric write your own check the
   worker never saw (finite-difference gradient checks, equivalence pins, decomposing headline
   benchmark aggregates). A tiny gap that is best-of-N or an aggregate over slices is suspect —
   demand noise bands.
2. Adjudicate contradictory reviews with verified numbers, not by averaging; write the decision
   down so it isn't relitigated.
3. Check `git status` every review; push the owner to commit; keep unrelated changes separate.
4. Specs must be source-grounded (re-grep line numbers, exact signatures) and land green per
   milestone. DISTRIBUTIONAL_HEADS_SPEC.md is the format precedent.

First moves: `git status && git log --oneline -8`; background `python -m pytest -q`; skim
KALMAN_READINESS_PLAN.md §W0 + the Kalman verdict (handoff §6) + the memory file. Then ask the
owner which thread is live: production DARKO replay (external, needs private data), linear
residual boosting (library-side, plan→spec), or PyPI/G1 polish.
```

## 11. Claude / Fable workflow specifics (account/app)

- **Model:** the outgoing supervisor was **Claude Fable 5**; the owner just switched the session to **`claude-opus-4-8`**, so the incoming supervisor is likely Opus. Same role and rules apply.
- **The Codex worker is not a tool you invoke** — it is a separate agent the owner drives. You receive its reports as pasted messages and verify them. Same for **Oracle reviews** (pasted external LLM reviews).
- **Persistent memory** (reload context fast, and update when state changes):
  `/Users/kmedved/.claude/projects/-Users-kmedved-Code-GitHub-chimeraboost/memory/chimeraboost-2026-07-review-findings.md` (indexed by `MEMORY.md` in that dir). It currently holds the full distributional + Kalman history incl. the round-2 replay verdict. This path is machine/user-specific; on a new machine the memory system regenerates its own dir, so the durable cross-machine record is *this handoff + the repo's `*.md` docs*.
- **Task chips** (`spawn_task`): create a chip the owner clicks to start a spinoff session in a fresh worktree. One is pending: `task_8057f8f7` (production DARKO replay). Chip ids don't persist across app restarts.
- **Scratchpad:** a session-specific dir is provided for throwaway files — use it, not `/tmp`. It does not travel.
- **Owner cadence:** the owner pastes a worker/Oracle report and asks "check this / what next / write a plan." They decide scope and timing; you provide verified judgment and the next concrete moves.

---
*Don't overclaim. Verify independently. Protect the working tree. Write down what you decide.*
