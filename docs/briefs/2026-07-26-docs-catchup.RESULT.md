# RESULT — documentation catch-up (v7.9x → v7.12x, ISSUES #10 and #15)

Branch `worker/docs-catchup`, pinned to current `master` at `984eb8a`.

## What was written and where

### VERSION_HISTORY.md

- **Phase 7 table**: four rows added (v7.9x, v7.10x, v7.11x, v7.12x) with
  headline A1/B1/N/features and one-line Change descriptions. v7.12x bolded as
  current champion.
- **Tags line**: extended with the four new tag hashes.
- **Four detailed sections** (v7.9x through v7.12x), each quoting recorded
  numbers from the git tag messages and the RESULT bundles:
  - v7.9x: continuation filter v2, dated-span demotion, frame 949, `is2019`
    collapse, the pure-removal protocol lesson.
  - v7.10x: `prev_cap_pct` repair (+0.0135 t=7.6), 2018 scrape coverage,
    2026 cap correction and the stale-in-lockstep failure mode, frame 944.
  - v7.11x: feature batch Arm C adopted at the boundary (+0.00178 t=1.96),
    adoption rationale, protocol correction (seed-paired t vs fold-paired t),
    rejected arms and why.
  - v7.12x: designated-ceiling award-path fix, max zone 56→68, zone MAE
    6.18→4.44.
- **Key milestones chart**: updated from v7.8x (0.765) to v7.12x (0.788).
  Narrative rewritten to cover the full v7.1x–v7.12x arc.

### METHODOLOGY.md

- **Hyperparameters section**: replaced the "re-sweep is the obvious next task"
  paragraph with the v7.8x retune findings — sigma monotone in zone MAE (the
  valve mechanism), gate_frac inert, sigma_left hook (ISSUES #10).
- **Acceptance rules**: added rule 4 (relative C1 calibration gate, |slope − 1|
  excess ≤ 0.005 vs incumbent) with the adjudication rationale (ISSUES #10).
  Added the pure-removal judging protocol from v7.9x (ISSUES #15).
- **Continuation rows section**: added the v7.9x dated-span replacement, the
  span rules (option-aware anchor, extension fallback, date-resolved blocks),
  and the renegotiation convention with its six pairs (ISSUES #15).
- **Stage 2 section**: added the designated-ceiling gate (v7.12x) and the
  standing decision that the Stage-2 clip never reads the observed salary
  (from QUEUE.md, decided 2026-07-26).
- **Data sources table**: updated `salaries_prehistory.csv` row count from 368
  to 767 (reflecting the v7.10x 2018 team-salary scrape). Updated the
  No-decrease floor paragraph similarly.

### ISSUES.md

- **Deleted #10** (sigma sweep mechanism + relative C1 gate → now in
  METHODOLOGY's hyperparameters and acceptance rules sections).
- **Deleted #15** (v7.9x conventions → now in VERSION_HISTORY and METHODOLOGY's
  continuation filter and acceptance rules sections).
- All other entries left untouched.

## Source conflicts found

None. The tag messages, RESULT bundles, landing commit messages, and ISSUES.md
entries all agree on the recorded numbers. The one non-obvious cross-reference:
the v7.10x tag says `A1 0.7849` and the feature-batch RESULT's incumbent reads
`A1 0.7849` — consistent (both on the same 944-row frame post-repair).

## Files touched

| file | change |
|---|---|
| `VERSION_HISTORY.md` | +4 table rows, +4 detailed sections, updated milestones |
| `METHODOLOGY.md` | hyperparameters (sigma), acceptance rules (C1 + pure-removal), continuation filter (spans + renegotiation), Stage 2 (designated ceiling + standing decision), data sources |
| `ISSUES.md` | deleted #10 and #15 |
| `docs/briefs/2026-07-26-docs-catchup.RESULT.md` | this file |
