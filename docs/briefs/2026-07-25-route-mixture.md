# Task brief: signing-route probability function + the max branch (phase 1 of the unified architecture)

Read `docs/worker-brief.md` first — isolation section, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.10x line, 944-row frame). Deliver a branch +
`docs/briefs/2026-07-25-route-mixture.RESULT.md`. A parallel worker may be
running the feature batch (`2026-07-25-feature-batch.md`) — do not pull
mid-task; your machinery is feature-agnostic and will be re-run with its
winners at landing.

## The architecture (decided; you build phase 1)

A multiclass probability function over signing routes plus per-route
structural values. Ex-ante output composes the branches by P; ex-post (route
told) collapses to the branch — the project's Stage-2 semantics. Branches
adopt ONE AT A TIME, each through its own gate. **This brief builds the
machinery and adopts (at most) the MAX branch.** Floor, MLE, and the
continuous per-route δ corrections are later phases on the same machinery.

Two disciplines are load-bearing:

1. **P is an OUTPUT composition weight, never a feature.** Nothing derived
   from the classifier enters `FEATURE_COLS`. (The rejected experiment in the
   ablation table — fold-honest P(mechanism|x) fed to the regression,
   −0.0073 — is input-side and does not veto this architecture; say so in the
   RESULT so the graveyard is not misread.)
2. **Classes are defined by where the salary LANDED**, not by Spotrac labels:
   `max` = `is_max_contract`; `floor` = `is_at_floor`; `mle` = year-1 pay
   within 2% of that season's exception amounts
   (`data/raw/raw_external/mle_exception_amounts.csv`); `continuous` =
   everything else. Spotrac's `signing_cat` corroborates, never defines.

## Part 1 — the classifier

Multiclass (4 classes above), fold-honest OOF: GroupKFold by player, the
suite's folds, 10 seeds averaged, XGBoost with modest capacity (the max class
has n=56). Features: current `FEATURE_COLS` (repaired `prev_cap_pct` is in).
Report per-class discrimination and calibration:

- P(max): AUC, calibration curve, and the P distribution on true max rows.
  Baselines to beat, from the 2026-07-24 diagnostic on the pre-repair frame:
  AUC 0.9595, median P on true maxes 0.356, 22 non-max rows above 0.5.
- P(mle): the separability AUC the MLE recon made this branch conditional on
  (its RESULT bounded the prize at +0.0143 oracle; whether P(mle) is
  learnable from player features — team-cap membership is invisible — is
  exactly what this number decides). REPORT ONLY; no MLE branch this phase.
- P(floor): report for phase 2.

## Part 2 — the max branch, push-then-clip

Primary form: adjusted latent = latent + P(max) × (1.05 × max_eligible_pct −
latent), then Stage 2 clips into [floor_pct, max_eligible_pct] unchanged.
High-P rows cross the ceiling and land exactly ON it; low-P rows are
untouched (median non-max P was 0.0002); the ambiguous middle moves partway.
The 1.05 margin is a FIXED constant — do not tune it on zone MAE (that
re-opens the one-way valve both sigma sweeps closed; if 1.05 fails to pin
high-P rows, report the pinned-share curve, do not optimize it).

Reference arms, reported alongside: (r1) mean-form mixture
P×ceiling + (1−P)×champion; (r2) hard switch at P>0.5. Ship form is
push-then-clip if it gates.

## Judging the max branch

Fixed v7.10x rows throughout (true-max zone n=56, reference MAE $5.48M;
counterweight band = non-max with cap_pct in [0.70, 0.90) of ceiling,
n≈44 — recompute the exact reference from the current suite before any run).

1. **Win**: true-max zone MAE improves ≥ $0.50M at 10 seeds. (Bar higher
   than the censor experiment's $0.30M: the global dial already proved
   −$0.26M is reachable while calibrated, and the oracle is −$4.8M — a
   per-row mechanism that cannot beat twice the global dial is not worth a
   structural change.)
2. **Brakes**: counterweight band signed bias growth ≤ +$0.30M (the censor
   result's instrument — that band is itself underpredicted $4.6M, so
   moderate lift is fine, inflation is not); 25%+ predicted-band bias stays
   within +$0.30M of the incumbent's (band table binding, global slope
   reported only — the oracle itself moves the slope).
3. **Guardrails**: ΔSel t > −2 (targeted intervention; pooled neutrality
   acceptable, improvement welcome); C2 fixed segments ≤ $0.3M growth; B1
   forward drop ≤ 0.003.
4. **Diagnostics that decide phase 2**: with the repaired features, does
   median P on true maxes move materially above 0.356? Which true maxes stay
   below P=0.3, and are they the Booker class (awaiting the feature batch)
   or irreducible?

## Traps

- The classifier's max label derives from the target's position — that is
  fine for a fold-honest OUTPUT weight and would be leakage only as an input
  feature. Keep the boundary clean in code (the P array never joins the
  feature frame).
- Class imbalance: 56/240/~90/~560. No scale_pos_weight games that wreck
  probability calibration — calibrated P is the whole point. Prefer modest
  depth and monotone-ish capacity; report the calibration curve.
- If the feature batch lands mid-task, do NOT rebase; finish on your pin.
- Do not touch sigma, gates, k_floor, or the censor population — the
  push-then-clip layer composes with the existing training loss unchanged.
- Version: if the max branch gates, the landing starts the v8.0 line — the
  tag and adoption are the architect's.

## Deliverable

Branch (machinery + inert integration behind a flag, default off) + RESULT:
classifier diagnostics (all classes), the three-arm max-branch table with
per-fold deltas on fixed rows, the brake/guardrail table, the pinned-share
curve, and the phase-2 recommendation. Expected effort: one to two days.
