# Task brief: route-mixture phase 2 — enriched classifier + precision-gated max branch

Read `docs/worker-brief.md` first — isolation section, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.11x line: 944-row frame, prev-season
`prev_cap_pct`). Deliver a branch +
`docs/briefs/2026-07-25-route-mixture-p2.RESULT.md`.

## Context you must not re-litigate

- Phase 1 (`2026-07-25-route-mixture.RESULT.md`): classifier good (P(max)
  AUC 0.9650, median on true maxes 0.5217, 13 non-max above 0.5), raw-P
  continuous push failed the brakes. The machinery is in
  `src/model/route_mixture.py`, flag default off.
- The 13 P>0.5 non-max rows are being re-reviewed row-by-row by a separate
  agent; the current working conclusion is that they are **genuine
  classifier false positives** (e.g. Reaves 2026 signed his largest legal
  deal, coincidentally ≈25% of cap — NOT a max). Therefore: the max CLASS
  definition (`is_max_contract`) stays exactly as-is, and you must NOT
  build tier-aware push targets that trust P. If that review lands mid-task
  it may reclassify rows; finish on your pin regardless.
- Global-dial ceiling is −$0.26M (censor-widening RESULT); the fixed 1.05
  push margin is frozen (never tuned on zone MAE).

## Part 1 — enrich the classifier

Retrain the 4-class classifier adding the delivered feature-batch columns
(`data/processed/feature_batch_columns.csv`: trend, signdelta +
`is_priced_early`, `est_value_prev`, `rookie_award_tier`,
`award_score_cum_clean`) as **classifier-only** inputs. They failed the
REGRESSION gates; the classifier is a different model with a different
target, and P never enters `FEATURE_COLS`, so no leakage path exists —
state this in the RESULT. Prune with sane feature selection if capacity
becomes a problem (56 positives).

Report vs the phase-1 baselines: P(max) AUC (0.9650), median P on true
maxes (0.5217), non-max above 0.5 (13), the calibration table, and the
**purity curve**: for a grid of thresholds τ, the empirical max-rate and
row counts among OOF rows with P ≥ τ. Also report P(floor) diagnostics
(AUC 0.8233 baseline) for phase 3, and P(mle) for the record.

## Part 2 — threshold by calibration purity

τ is chosen from the PURITY CURVE, fixed before any zone metric is
computed: **the smallest τ whose OOF purity (empirical max-rate at P ≥ τ)
is ≥ 90%**. If no τ ≤ 0.95 reaches 90% purity, that is the result — report
it and stop at Part 1; the branch waits for a better classifier, and
"stop" is a legitimate deliverable (the phase-1 [0.7,0.9) bin was only 72%
pure, so this outcome is live).

## Part 3 — the gated max branch

Push-then-clip, gated: `adj = latent + P·(1.05·ceiling − latent)` for rows
with P ≥ τ; latent untouched below τ. Report the hard variant
(pred = ceiling for P ≥ τ) alongside. Judged on FIXED v7.11x rows
(recompute references from the current suite before any run):

1. **Win**: true-max zone MAE improves ≥ $0.50M at 10 seeds.
2. **Brakes**: counterweight band ([0.70,0.90) of ceiling, non-max) signed
   bias growth ≤ +$0.30M; 25%+ predicted-band bias growth ≤ +$0.30M.
3. **Guardrails**: ΔSel t > −2; C2 fixed-segment |bias| growth ≤ $0.3M;
   B1 forward drop ≤ 0.003.
4. Report the collateral list outright: every non-max row above τ, with its
   push damage — the review agent's classification will be checked against
   exactly this list.

## Traps

- τ from purity only; never from zone MAE (three experiments have died on
  that altar — the protocol block in `evaluate_suite.py` records why).
- The failed-arm columns are classifier inputs ONLY; nothing enters
  `FEATURE_COLS`.
- Class labels stay `is_max_contract` as-is; no tier widening, no
  relabeling, pending the external row review.
- Flag stays default-off on your branch; adoption/tag (v8.0 line) is the
  architect's.
- Expectation management is part of honesty: with a high-purity τ the
  branch may touch only ~15-25 rows. Compute the honest win ceiling for
  your τ (sum of current champion errors on the touched true maxes) before
  running, and report it next to the realized win.

## Deliverable

Branch + RESULT: enriched-classifier diagnostics vs phase-1 baselines, the
purity curve and chosen τ, the three-arm gated table with per-fold deltas,
the brake/guardrail battery, the touched-row collateral list, and a phase-3
recommendation (floor branch go/no-go on the enriched P(floor)). Expected
effort: half a day to a day.
