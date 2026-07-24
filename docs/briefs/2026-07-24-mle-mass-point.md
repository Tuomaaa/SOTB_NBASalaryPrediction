# Task brief: MLE mass-point reconnaissance (measure only)

Read `docs/worker-brief.md` first. This is a half-day MEASUREMENT task in the
AAV-memo mold: you produce numbers and a recommendation; you change no table,
no filter, no model, and claim no version number.

**Pin**: current `master` (v7.9x line) in your own worktree. A concurrent
worker is modifying `prev_cap_pct` on its own branch — irrelevant to you; do
not pull anything mid-task.

## Why

The model treats the CBA's max and minimum as structure (censoring + Stage-2
clip) because prices pile up ON those numbers. The mid-level exceptions are
also CBA-fixed dollar amounts per season — full (non-taxpayer) MLE, taxpayer
MLE, room MLE, and the bi-annual exception — and many contracts sign at
exactly those numbers. If that pile-up is thick, the mixture-output design
(tasks #4/#8) gets a third branch for free; if it is thin, we close the
question with numbers. MLE-labeled rows on the 949 frame: n=124, bias
+$0.83M, MAE $2.72M (suite C2 table).

## What to measure

1. **The exception amounts per season, 2019-2026, with sources.** Two
   independent routes, both required, they must agree:
   - *Empirical*: spike-hunt the year-1 salary distribution per season —
     mass points reveal themselves (this is how `floor_pct` recovers the
     minimum scale; copy that pattern).
   - *Published*: the amounts as listed by a public source (Spotrac/RealGM
     CBA pages), hand-curated into a small CSV under
     `data/raw/raw_external/` with a source URL column. Amounts are
     load-bearing the same way cap values are — a wrong number silently
     misclassifies, so document where each came from.
   Convert to cap_pct per season before any comparison; dollars are display
   only.
2. **Pile-up thickness.** Evaluation rows (949 frame) whose year-1 pay sits
   at each exception amount: exact, ±1%, ±2%. Split by `signing_cat` (the
   label is diagnostic-only and ~10% Unknown — detect by salary proximity,
   use the label as a cross-check, never as the filter). Report per season
   and per exception type.
3. **Does the model already price them?** Champion OOF bias/MAE on the
   at-exception rows vs the rest of the $3-15M band.
4. **The oracle bound** (mirror of the max-zone oracle): set prediction =
   the exception amount for rows AT it; report band MAE, pooled R², and the
   calibration effect. That is the ceiling on what a snap/branch could ever
   be worth.
5. **The false-positive cost surface**: rows within ±2% of an exception that
   are NOT exception signings (cap-space deals landing there by coincidence)
   — count them; they are who a snap rule would hurt.

## Traps

- Exception amounts changed definition at the 2023 CBA — verify per-era, do
  not extrapolate one season's ratio-to-cap across the boundary.
- Multi-year MLE deals escalate off the year-1 amount; only year 1 sits on
  the mass point. The frame is year-1 rows, so this should be automatic —
  confirm it.
- `signing_cat` mislabels exist (six >$6M "Minimum" rows are documented in
  ISSUES #4); salary proximity is the instrument, labels corroborate.

## Deliverable

`docs/briefs/2026-07-24-mle-mass-point.RESULT.md` per the evidence-bundle
format: the amounts table with sources, thickness counts, current-model
errors, the oracle numbers, and a one-paragraph recommendation — third
mixture branch worth designing (with the expected prize) or question closed.
Expected effort: half a day.
