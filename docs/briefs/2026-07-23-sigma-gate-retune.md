# Task brief: retune sigma and both censoring gates (ISSUES #7)

Read `docs/worker-brief.md` first — especially the isolation section, and cap
your threads: this task is grid-heavy and other sessions share the machine.

**Pin**: tag `v7.8x` in your own worktree. Deliver evidence + (only if you
extend the objective, see below) a small branch.

## Why

The champion's censoring settings were tuned on a row set that no longer
exists: sigma=0.02 and gate_frac=0.55 date from the 1,487-row era with the
buggy ceilings; k_floor=2.0 got a three-point screen only (1.5/2.0/3.0). The
row set is now 1,172, the ceilings are audited, and the correct judging
instruments exist — the two zone scorecards. ISSUES #7 has the full context.

## Search space

- `sigma` ∈ {0.01, 0.02, 0.04} (shared both sides first pass)
- `gate_frac` (right) ∈ {0.45, 0.55, 0.65}
- `k_floor` (left) ∈ {1.5, 2.0, 2.5, 3.0}
- Optional second pass: per-side sigma. `_make_tobit_obj` currently takes one
  sigma; extending it with a `sigma_left` parameter is an additive change you
  may include on your branch for the architect to review — do not change the
  default behavior.

Screen the grid at 3 seeds; confirm the top ~3 configs at 10 seeds. Fitters:
`make_grabit_fitter(sigma, gate_frac, floor_gate_k)` in
`src/model/evaluate_suite.py` already takes all three.

## Judging (this is the part that must not be improvised)

A config wins on the ZONE scorecards, with pooled metrics as guardrails —
never the reverse:

1. **Zone objectives**: right-zone MAE (n=57, currently $6.09-6.14M, 55/2
   better/worse) and floor-zone MAE (n=297, currently $1.98M, ~255/40).
   A candidate must improve at least one zone by ≥ $0.10M without degrading
   the other by more than $0.05M.
2. **Guardrails**: selection-pool paired ΔSel vs the v7.8x reference
   matrices not significantly negative (t > -2); C1 calibration slope stays
   in [0.99, 1.01]; C2 fixed-row mechanism segments — no |bias| growth over
   $0.3M (fixed rows, not predicted bands).
3. **B1 sanity** on the final config only: forward 2024-26 must not drop by
   more than 0.003.

Reference numbers and matrices: `outputs/models/evaluation_suite.json`
(`fold_r2_selection`) and `oof_reference.csv` at the pinned tag — copy them
into your worktree before any suite run of your own overwrites them (the
suite keeps one `_prev` generation, but belt and suspenders).

## Known traps

- Zone membership moves with the gates: a looser gate censors more rows, and
  MAE over a DIFFERENT zone population is not comparable. Report every
  config's zone MAE over the FIXED v7.8x zone rows (n=57 / n=297) as the
  headline, and the config's own-zone numbers as context
- sigma trades CV against holdout historically (METHODOLOGY's old note);
  the paired guardrail is the arbiter, not holdout eyeballing
- 3-seed screen SEs are ~2x wider — do not eliminate configs within one SE
  of the leader at screen stage

## Deliverable

`docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md` per the evidence-bundle
format: full grid table (config x zone MAEs x guardrails), the confirmed
winner with 10-seed numbers, per-fold deltas, and your one-sentence mechanism
explanation for WHY the winner wins (a config that wins without an explanation
is an anomaly, not a result). If the winner is the incumbent
(0.02/0.55/2.0), say so plainly — "already optimal" is a fine outcome and
closes ISSUES #7 either way. Expected effort: half a day.
