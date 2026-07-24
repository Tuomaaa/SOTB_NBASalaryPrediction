# Task brief: AAV target-variable feasibility memo

Read `docs/worker-brief.md` first. This is an ANALYSIS task — no pipeline
changes, no model adoption. Your deliverable is a memo; the go/no-go decision
it informs belongs to the project owner because switching targets breaks
comparability with every published version number (a v8.0-scale break, same
nature as the v3.5x→v4.0 filter break in VERSION_HISTORY).

**Pin**: tag `v7.8x` (the current champion; `git checkout v7.8x` to reproduce
its numbers, work on a branch off it).

## Question

The model predicts `cap_pct` of the contract's FIRST year. But "market value"
is arguably the whole contract: front-loaded and back-loaded deals put
different year-1 numbers on identical total commitments, injecting noise the
model cannot learn. Would AAV (average annual value, as % of the signing-year
cap) be a better target?

## What exists

- `data/processed/spotrac_signing_types.csv` — 8,910 rows with `aav`,
  `total_value`, `contract_years` per contract, rebuilt 2026-07-23 with
  anchored season spans. Salary-aware matching helper:
  `attach_signing_labels` in `scripts/diagnostics.py` (mind its docstring —
  one season can hold two contracts after a buyout)
- `contract_years` also lives in the training table itself
  (`data/processed/training_data_v2.csv`)
- The evaluation frame builder: `load_evaluation_frame` in
  `src/model/evaluate_suite.py` (1,172 rows post-filters at v7.8x)

## Required analyses

1. **Coverage**: what share of the 1,172 evaluation rows can be assigned a
   trustworthy AAV? Match Spotrac contracts salary-aware; report coverage
   overall, by season, by salary band. State the AAV-vs-year-1 tolerance you
   used to call a match trustworthy.
2. **Divergence**: distribution of AAV/cap vs year-1/cap — how many rows
   differ by >2%, >5%, >10% of themselves? Which contract types diverge
   (long deals? max deals? declining-structure deals)? Name the 10 largest
   divergences and sanity-check two by hand against Spotrac's site numbers.
3. **Bounds interaction**: Stage 2 clips into
   `[floor_pct, max_eligible_pct]` — both defined on YEAR-1 pay. Write down
   how each bound would have to be restated for an AAV target (a max deal's
   AAV exceeds its year-1 tier % because raises are on year-1 salary; the
   floor has the mirror issue). This section decides whether the switch is
   mechanically clean or a Stage-2 redesign.
4. **Trial fit** (evidence, not adoption): on the covered subset, refit the
   champion with AAV/cap as target. R² against the year-1 target is NOT
   comparable (different target variance — D1 logic applies to targets too);
   instead report (a) target's own noise proxy: sd of |AAV − year1| within
   matched rows, (b) whether OOF residual patterns (C1 calibration shape, C2
   mechanism biases) look structurally cleaner or dirtier, (c) the same
   comparison restricted to rows where AAV == year-1 (the null-difference
   subset, as a sanity anchor).
5. **Recommendation**: one of — switch (v8.0), dual-target (report both,
   publish year-1), or reject; with the two strongest reasons and the main
   risk of your recommendation.

## Traps known in advance

- Spotrac AAV for renegotiated/extended deals reflects the NEW money; the
  year-1 row may belong to the superseded shell (ISSUES #2's warning)
- 2019 rows: weakest Spotrac coverage; do not let 2019 gaps masquerade as a
  finding about AAV itself (the season-alignment lesson in the worker brief)
- Escalator direction flipped in some eras (declining deals allowed 8% down);
  do not assume AAV ≥ year-1

## Deliverable

`docs/briefs/2026-07-23-aav-target-memo.RESULT.md` following the evidence-
bundle format in the worker brief (sections 1, 2, 4, 5, 7 apply; there are no
gate verdicts because nothing is being adopted). Expected effort: half a day.
