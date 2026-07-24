# Task brief: supervised widening of the right-censor gate

Read `docs/worker-brief.md` first — isolation section, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.9x line). Two other workers are out on their own
branches (prev_cap_pct, MLE recon) — do not pull mid-task. Deliver a branch +
`docs/briefs/2026-07-24-censor-widening.RESULT.md`.

## The idea (user-proposed) and why it is now admissible

The sigma sweep (docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md) closed
"tune sigma on the max zone" because that zone has no counterweight: every row
is at its bound, any push scores, zone MAE is monotone in sigma out to the
grid edge. **This experiment changes the censored POPULATION instead**: also
treat near-max rows — paid at least `c` of their own ceiling but below the
max flag — as right-censored. Those rows' salaries are true market prices, so
over-pushing lifts predictions past them and their error RISES. The widened
population contains its own brake; sigma acquires an interior optimum and
becomes tunable. Higher sigma than 0.02 is now on the table BECAUSE the brake
exists — that is the point, not a violation of the closed sweep.

Known cost to keep in view: inside the LOSS, a widened row still contributes
only "worth at least X", discarding the top half of its price information.
The tuning instrument supervises sigma globally; it does not restore that
per-row information. Whether the trade nets out is what the grid answers.

## Search space

- `c` (censor-inclusion threshold, fraction of own `max_eligible_pct`) ∈
  {0.70, 0.75, 0.80, 0.85}; today's behavior ≈ the `is_max_contract` gate at
  0.90.
- `sigma` (right side only) ∈ {0.02, 0.03, 0.04, 0.06} — pass `sigma_left`
  explicitly pinned at 0.02 so the floor side is untouched (the hook shipped
  at the sigma retune, verified inert when unset).
- Left gate `k_floor` = 2.0 fixed. Champion hyperparameters otherwise.

Implementation: the right-censor mask in `train_grabit` /
`make_grabit_fitter` currently keys on `is_max_contract`; parameterize the
mask as `cap_pct >= c * max_eligible_pct` (a training-time quantity, fine on
training folds). Keep `is_max_contract` itself untouched — the zone scorecard
definition does not move.

Copy the sigma-retune harness: 3-seed screen over the 16 configs, 10-seed
confirmation of the top ~3. Report every config's gated-row count (gR).

## Judging

All zone/segment rows FIXED from the v7.9x reference before any run:

1. **True-max zone** (`is_max_contract`, n=56): MAE and the pinned-at-ceiling
   share. This is what the experiment exists to improve — current reference
   $6.30M (suite, 10-seed).
2. **The counterweight band** (non-max rows with cap_pct in [0.70, 0.90) of
   ceiling — report n): bias and MAE. Over-pushing shows up here first; bias
   growing past +$0.3M kills the config (C2's yardstick).
3. Guardrails: selection-pool paired ΔSel vs same-run incumbent, t > −2;
   C2 fixed mechanism segments |bias| growth ≤ $0.3M; **calibration judged on
   the predicted-band bias table, not the global slope** — a targeted top-end
   push moves the global OLS slope even when correct (the max-mixture oracle
   moved it 0.988 → 0.935; recorded in task #4's notes), so the slope is
   reported but the band table is the binding instrument.
4. B1 forward sanity on the final config only: no drop > 0.003.

Win condition: true-max zone MAE improves ≥ $0.30M at 10 seeds with the
counterweight band's bias growth ≤ $0.3M and guardrails clean. "No config
clears it" is a legitimate result and pins down how much a global dial can
buy — the per-row version (task #4) then owns the rest.

## Traps

- The widened set is defined off `cap_pct` (the target) — fine as a
  TRAINING-loss mask (same as the existing gates), but never as a feature and
  never recomputed on test rows.
- Zone membership must not move with `c`: all reporting on the FIXED v7.9x
  row lists above, config's own-gate counts as context only.
- The interior optimum may be shallow; 3-seed SEs are ~2x wide — do not
  eliminate configs within one SE at screen stage (sigma-retune rule).
- A same-run incumbent column in every table; never compare against stored
  headlines.

## Deliverable

RESULT per the evidence-bundle format: full grid (config × true-max MAE ×
pinned share × counterweight bias × ΔSel × band table), confirmed winner or
"no winner" with per-fold deltas, and the one-sentence mechanism story for
whichever way it goes. Expected effort: half a day.
