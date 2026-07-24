# Task brief: clear the correctness debt (ISSUES #3, #17, #18 + predict.py)

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-26-cleanup-debt.RESULT.md`. Three other workers may be
out (service years #20, route-mixture phase 3, route δ) — do not pull
mid-task. **You own no file they own**: they work in `train.py`,
`route_mixture.py` and the ceiling rules; you work in the award/name joins,
`predict.py` and `grabit_results.json`. If you find yourself editing
`_compute_max_eligible` or `route_mixture.py`, stop and escalate.

Four independent items. Each is small; the discipline is that **two of them
change a model input**, so they are measured, not assumed.

## Item 1 — ISSUES #18: the awards name join drops footnote-marked stars

90 of 943 rows in `awards_full.csv` carry footnote junk in `player_name_norm`
(`^ § † digits`, replacement chars) that `build_external_features.norm` does
not strip, so `award_score_cum` / `all_nba_cum` silently lose award seasons
for the biggest names — **every one of the 13 ROY winners**, plus MVP/All-NBA
rows for Durant, Curry, Giannis, Jokić, Harden, Westbrook.

Fix the normaliser (strip footnote marks and non-name codepoints before the
join), rebuild the external features, and verify: `award_score_cum > 0` for
Luka/Ja in their post-rookie seasons; the 13 ROY winners all carry their
award; no name that previously joined stops joining.

**This changes a feature's values, so it is measured**: paired ΔSel, 10 seeds
× 5 folds, **per-fold** pairing (the fold is the unit — a seed-paired t
inflates ~3×, see the feature-batch RESULT §4). The feature batch already
scored this fix in isolation at **ΔSel −0.00422 (t −1.67)** — expect a small
negative. **Adopt anyway unless the delta is significantly negative
(t < −2)**: the recovered award mass is redundant with `darko`/`prev_cap_pct`
and the affected players are ceiling-pinned, so it adds a little variance
without lift — but the table is objectively wrong and a wrong input is a
wrong input. Precedent: v7.9x was CV-neutral and adopted for correctness.
Report the delta plainly either way and let the architect make the call if
t < −2.

## Item 2 — ISSUES #17: the Kanter → Freedom rename

`salaries_prehistory.csv` misses ~6 of the 2019 evaluation rows; one is a pure
rename — Enes Kanter's 2018-19 pay is on the page under the old name while
the training data carries `enes freedom`. Add a small, documented alias map
(a CSV under `data/raw/raw_external/`, not a hard-coded dict) applied in
`scrape_2018_salaries.py` and `_load_prev_season_cap_pct`, and check for
other post-2019 renames in the same sweep (report what you find; there are
few and they are known — Ron Artest-era renames are outside our window).

Verify: `_load_prev_season_cap_pct()[("enes freedom", 2018)]` returns a value.
Measured the same way as item 1 (it moves `prev_cap_pct` on at least one
row). The two-way/minimum players in #17 are NOT in scope — they need a
two-way salary source, not an alias.

## Item 3 — ISSUES #3: two different numbers called "the CV R²"

`train.py` writes `grabit_results.json` from a **single-seed** run while every
document quotes the **10-seed** average. Make `train.py` write the 10-seed
average (the canonical one) into that file, keeping any single-seed value
under an explicitly different key if it is still wanted. No model change; no
delta to measure. Verify the file's headline matches
`evaluation_suite.json`'s A1 to 4 dp.

## Item 4 — `src/model/predict.py` is an orphan

`CLAUDE.md` documents it as "inference on upcoming free agents", but it ships
**plain XGBoost with no censoring and no Stage-2 clip** — the champion stack
lives in `evaluate_suite.py` and `scripts/export_web.py` (which does use
`train_grabit` + `_compute_max_eligible` + the clip). The next agent will
read `predict.py` and believe it is production.

Rewire it to call the same path `export_web.py` uses, so the documented entry
point actually predicts what the project claims to predict. If that turns out
to be a bigger job than a delegation wrapper, say so and instead make the file
fail loudly with a pointer, rather than silently serving a different model.
Do NOT edit `CLAUDE.md` (architect/docs lane) — state in the RESULT what it
should say.

## Traps

- Items 1 and 2 change model inputs; items 3 and 4 do not. Do not report a
  delta for 3 or 4, and do not skip it for 1 and 2.
- Rebuilding the training table churns cosmetic `team`/`agent`/`position`
  columns (ISSUES #5's known non-determinism) — apply the stage that owns
  your change in isolation and verify **only the intended columns move**, the
  way the prev-cap and feature-batch workers did.
- The 10-seed rebuild is the expensive part; budget for it.

## Deliverable

Branch + RESULT: per-item what changed, the two measured deltas with per-fold
values, the verification outputs, and an explicit list of any file you
touched that another in-flight worker also touches (there should be none).
Expected effort: half a day to a day.
