# RESULT: clear the correctness debt (ISSUES #3, #17, #18 + predict.py)

**Branch**: `worker/cleanup-debt` off `master` at `984eb8a`

---

## Item 1 — ISSUES #18: awards name join drops footnote-marked stars

### What changed and why

`scripts/build_external_features.norm()` did not strip footnote junk from
`awards_full.csv`'s `player_name_norm` column — trailing `^`, `§`, `†`, the
Unicode replacement character, and concatenated annotation tokens (`st1`,
`covid2`, etc.). 83 of 943 award rows carried these, covering every ROY winner
from 2015 onward plus MVP/All-NBA seasons for Durant, Curry, Giannis, Jokic,
Harden, Westbrook, and others. The join to training names missed them silently.

Two changes:
1. `norm()` now strips concatenated annotation suffixes (`st\d+`, `covid\d+`)
   and then removes all non-name characters via `re.sub(r"[^a-z .'-]", "", out)`.
2. `build_award_features()` applies `norm()` to the CSV's `player_name_norm`
   column before aggregation, so duplicate dirty names (e.g. `joel embiid^` +
   `joel embiidst1` + `joel embiidcovid1`) merge into one player before the
   join. Without this, the main() merge produced row duplication (3113 → 3239).

**Effect on training data**: 148 rows gained recovered award mass in
`award_score_cum`. `all_nba_cum` was unchanged (the All-NBA rows were `^`-marked
names that the pn_clean join partially caught). No other columns moved.

### Numbers table

| Metric | Reference (pre-fix) | Candidate (post-fix) | Δ |
|--------|--------------------:|---------------------:|--:|
| A1 CV R² | 0.7885 | 0.7877 | −0.0009 |
| A2 CV R² (2024-26) | 0.8336 | 0.8349 | +0.0013 |
| B1 origin 2024 R² | 0.8580 | 0.8606 | +0.0026 |
| B1 origin 2025 R² | 0.8222 | 0.8232 | +0.0011 |
| B1 origin 2026 R² | 0.7903 | 0.7948 | +0.0045 |

**Paired ΔSel**: −0.00076 ± 0.00106 (SE), **t = −0.72**

Per-fold: [−0.00390, −0.00139, +0.00197, +0.00123, −0.00173]

### Verdict

t = −0.72, well above the −2 threshold. The brief pre-registered "adopt anyway
unless t < −2" because the recovered award mass is redundant with `darko` /
`prev_cap_pct` and the affected players are ceiling-pinned (variance without
lift). The table is now objectively correct. B1 forward moves favorably at all
three origins.

### Verification

- `norm("Kevin Durant^")` → `"kevin durant"` ✓
- `norm("bam adebayost1")` → `"bam adebayo"` ✓
- `norm("joel embiidcovid1")` → `"joel embiid"` ✓
- `award_score_cum > 0` for `luka doncic` in seasons 2021-2026 ✓
- `award_score_cum > 0` for `ja morant` in seasons 2022-2026 ✓
- 12 of 13 ROY winners (2015-2026) now carry their award ✓
  (Wiggins 2015 ROY is before AWARD_MIN_YEAR=2017; Flagg 2026 ROY appears only
  in season 2027+ due to lagging — both by design, not the fix)
- No previously-joining name stopped joining ✓
- No row expansion (3113 → 3113) ✓

---

## Item 2 — ISSUES #17: the Kanter → Freedom rename

### What changed and why

`salaries_prehistory.csv` carries Enes Kanter's 2018-19 salary under `enes
kanter`, but the training data carries him as `enes freedom` (legal name change
2021). The join in `_load_prev_season_cap_pct` missed, so his `prev_cap_pct` for
season 2019 fell back to the rookie-scale fill (0.073) instead of his true prior
salary (0.189, 18.9% of cap — a $9M veteran deal).

A second rename was found in the same sweep: `wesley iwundu` in prehistory vs
`wes iwundu` in training data (BBRef name variant).

Created `data/raw/raw_external/player_name_aliases.csv` (a documented alias map,
not a hard-coded dict) and applied it in:
- `src/model/train._load_prev_season_cap_pct()` via new `_load_name_aliases()`
- `scripts/scrape_2018_salaries.py` via `_load_aliases()` for future re-scrapes

**Effect on training data**: exactly 2 rows changed in `prev_cap_pct`:
- enes freedom 2019: 0.072973 → 0.189224 (+0.116)
- wes iwundu 2019: 0.014848 → 0.013530 (−0.001)

No other columns moved.

### Numbers table

Combined with Item 1 (see table above). The delta is dominated by Item 1's 148
award rows; Kanter alone moves 1 row's prev_cap_pct on a ceiling-pinned player.

### Verification

- `_load_prev_season_cap_pct()[("enes freedom", 2018)]` = 0.189224 ✓
- `_load_prev_season_cap_pct()[("wes iwundu", 2018)]` = 0.013530 ✓
- Old names (`enes kanter`, `wesley iwundu`) no longer appear in the lookup ✓

### Other post-2019 renames found

The remaining 9 season-2019 training players without prehistory-2018 matches
are two-way/minimum players (Caruso, Milton, Iwundu, Gabriel, Jefferson) or
players with no BBRef team-salary entry (Macura, Chealey, J. Williams, Dozier,
Ulis). These need a two-way salary source, not an alias — confirmed as out of
scope per the brief.

---

## Item 3 — ISSUES #3: two different numbers called "the CV R²"

### What changed

Added `_multiseed_grabit_cv()` to `train.py`: a 10-seed × 5-fold Grabit CV loop
that computes the seed-averaged pooled R², matching the protocol in
`evaluate_suite.oof_groupkfold`. The `__main__` block now calls this and writes
the 10-seed average into `grabit_results.json` under `cv_r2_mean`. The
single-seed value is preserved under `cv_r2_single_seed`, and `n_seeds` records
that the headline is a 10-seed average.

No model change. No delta to measure.

### Verification

Not yet run against `evaluation_suite.json` (no suite output exists in the
worktree). The architect should verify that after a fresh suite run, the file's
`cv_r2_mean` matches `evaluation_suite.json`'s A1 to 4 dp.

---

## Item 4 — predict.py rewired to the champion stack

### What changed

`predict.py` was using plain XGBoost (`train_xgboost`) with no censoring and no
Stage-2 clip. Rewrote it to use the same pipeline as `export_web.py`:
- `train_grabit()` for the two-sided censored-normal loss
- `_compute_max_eligible()` for CBA tier ceilings
- `_compute_floor()` for CBA minimum floor
- `np.clip(latent, floor_pct, max_eligible_pct)` for the Stage-2 bound

The script now trains on all seasons before the target (same as export_web.py's
forward mode), applies the full CBA clip, and reports `is_capped`/`is_floored`
flags alongside predictions. Feature borrowing from history follows the same
pattern as export_web.py.

No model change. No delta to measure.

### What CLAUDE.md should say

The architect should update CLAUDE.md's architecture table entry for
`predict.py` from "inference on upcoming free agents" to something like:
"inference on upcoming free agents — Grabit + Stage-2 CBA clip, same pipeline
as export_web.py". (I am not editing CLAUDE.md per the worker brief.)

---

## Files touched

| File | Item | What changed |
|------|------|-------------|
| `scripts/build_external_features.py` | 1 | `norm()` strips footnote marks; `build_award_features()` applies norm before aggregation |
| `data/raw/raw_external/player_name_aliases.csv` | 2 | NEW — alias map (2 entries) |
| `src/model/train.py` | 2, 3 | `_load_name_aliases()`, `_load_prev_season_cap_pct()` applies aliases; `_multiseed_grabit_cv()` + updated `__main__` |
| `scripts/scrape_2018_salaries.py` | 2 | `_load_aliases()` applied to scraped names |
| `src/model/predict.py` | 4 | Full rewrite: plain XGBoost → Grabit + Stage-2 clip |
| `data/processed/training_data_v2.csv` | 1, 2 | 148 rows: `award_score_cum` gained; 2 rows: `prev_cap_pct` corrected |

**No file touched by another in-flight worker** (they own `train.py`'s
`_compute_max_eligible`, `route_mixture.py`, and the ceiling rules; I touched
`_load_prev_season_cap_pct`, `_multiseed_grabit_cv`, and `__main__`).

## Proposed commit message

```
Fix: awards name join (#18), Kanter alias (#17), 10-seed CV R² (#3), predict.py to Grabit

Item 1: norm() in build_external_features now strips footnote marks (^, st1,
covid2, etc.) from awards_full.csv names before the join. 148 training rows
recover award mass for Durant, Curry, Giannis, Jokic, and all ROY winners.
Paired ΔSel −0.00076 (t=−0.72) — neutral as expected, adopted for correctness.

Item 2: player_name_aliases.csv maps enes kanter → enes freedom and
wesley iwundu → wes iwundu. Applied in _load_prev_season_cap_pct and
scrape_2018_salaries.py. 2 rows corrected.

Item 3: train.py writes the 10-seed average CV R² into grabit_results.json
(the canonical number) instead of a single-seed value.

Item 4: predict.py rewired from plain XGBoost to the champion Grabit +
Stage-2 CBA clip pipeline, matching export_web.py's inference path.
```

## ISSUES.md amendments

ISSUES #3, #17, #18 are fixed by this branch. The architect should delete them
from ISSUES.md when landing.
