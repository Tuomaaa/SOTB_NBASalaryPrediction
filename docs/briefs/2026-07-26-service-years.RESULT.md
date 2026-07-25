# RESULT: real service years for the tier ceilings (ISSUES #20)

**Branch**: `service-years` (worktree `BBall-worker-service-years`)
**Pin**: master at `e98ea82` (v7.12x line, 944 rows, max zone n=68)

## 1. What changed and why

`_compute_max_eligible` derived service years as `season - draft_year`, falling
back to `age - 19` for the 24% of rows without a draft entry. That fallback
systematically overstated service for undrafted and late-entry players, pushing
them into higher CBA tiers. Two rows were provably mis-tiered (Reaves 2026 and
Butler 2019), both sitting exactly on a tier boundary and both reading
`is_max_contract = False` when they are real maxes.

The fix adds a per-player debut season scraped from BBRef's player index pages
(`/players/{letter}/`, the same source as `height.py`). The "From" column gives
the end-year of each player's first NBA season; subtracting 1 converts to our
start-year convention. This covers 789 of 793 training-data players (99.5%).
The remaining 4 (jeff dowtin, jeenathan williams, luca vildoza, thomas sorber)
fall back to `age - 19`; only 1 of these (jeff dowtin) survives into the
944-row evaluation frame.

Butler 2019 required a second fix: with correct service years (8), his base
tier drops from 35% to 30%, but the All-NBA award path (`supermax`) re-grants
35%. Butler changed teams (PHI to MIA) and is not eligible for the Designated
Veteran Extension. Adding him to `designated_ineligible.csv` mirrors the
existing team-change entries (Kawhi 2019, Kyrie 2019, Kemba 2019). The brief's
trap warns against redesigning the supermax mechanism; this is a legitimate
data entry into the curated list, not a mechanism change.

## 2. Numbers table

### Paired delta (new vs reference, 10 seeds x 5 folds)

| Metric | Reference | Candidate | Delta |
|--------|-----------|-----------|-------|
| A1 CV R2 | 0.78778 | 0.78778 | +0.00000 |
| A2 CV R2 2024-26 | 0.83354 | 0.83364 | +0.00010 |
| B1 Forward R2 | 0.83035 | 0.83053 | +0.00018 |
| C1 Cal. slope | 0.98492 | 0.98430 | -0.00063 |
| Max zone n | 68 | 70 | +2 |
| Max zone MAE | $4.44M | $4.40M | -$0.04M |

**Paired ΔSel**: +0.00001 +/- 0.00007 (SE), t = +0.08
**Per-fold deltas**: [-0.00014, +0.00016, -0.00012, -0.00003, +0.00016]

This is a correctness fix, not a feature. The paired delta is essentially zero
as expected: the change moves 2 rows into the max zone and adjusts base-tier
ceilings for 119 rows, but the ceiling is not a model feature — it affects
only the censoring mask (2 rows) and the Stage-2 clip.

## 3. Invariant checks

All pass:

| Check | Expected | Actual |
|-------|----------|--------|
| Reaves 2026 tier_ceiling | 0.250 | 0.250 |
| Reaves 2026 is_max | True | True |
| Butler 2019 tier_ceiling | 0.300 | 0.300 |
| Butler 2019 is_max | True | True |
| Max zone n | 70 | 70 |
| Frame size | 944 | 944 |
| Over-ceiling rows | 0 | 0 |
| Exact-tier rows (all 53 ceiling == landed tier) | True | True |
| Mislabel filter drops | 6 | 6 (same players) |
| Continuation filter drops | 347 | 347 |
| Age-19 fallback rows (eval frame) | minimal | 1 (jeff dowtin) |

## 4. Ceiling movement table

119 base-tier movements in the evaluation frame: **117 down** (overstated
service corrected), **2 up** (understated service corrected).

### Top 10 by salary (all downward corrections)

| Player | Season | Exp old->new | Base old->new | Ceiling now | Pay |
|--------|--------|-------------|---------------|-------------|-----|
| Austin Reaves | 2026 | 8->5 | 0.30->0.25 | 0.25 | $41.2M |
| Jimmy Butler | 2019 | 10->8 | 0.35->0.30 | 0.30 | $32.7M |
| Isaiah Hartenstein | 2024 | 7->6 | 0.30->0.25 | 0.25 | $30.0M |
| Fred VanVleet | 2025 | 11->9 | 0.35->0.30 | 0.30 | $25.0M |
| Alex Caruso | 2025 | 11->8 | 0.35->0.30 | 0.30 | $18.1M |
| Allen Crabbe | 2019 | 7->6 | 0.30->0.25 | 0.25 | $17.8M |
| Duncan Robinson | 2025 | 11->7 | 0.35->0.30 | 0.30 | $16.8M |
| Ricky Rubio | 2019 | 10->8 | 0.35->0.30 | 0.30 | $16.2M |
| Duncan Robinson | 2021 | 7->3 | 0.30->0.25 | 0.25 | $15.6M |
| Marcus Morris | 2019 | 10->8 | 0.35->0.30 | 0.30 | $15.0M |

Sanity notes: Reaves is undrafted (debut 2021-22); Hartenstein and Crabbe were
undrafted and drafted-but-stashed respectively; Duncan Robinson went undrafted
and debuted 2018-19 (age-19 overstated by 4 years). All corrections are in the
expected direction: the age-19 proxy overstated service for late-entry and
international players.

### 2 upward corrections

| Player | Season | Exp old->new | Base old->new | Pay | Note |
|--------|--------|-------------|---------------|-----|------|
| Michael Kidd-Gilchrist | 2019 | 6->7 | 0.25->0.30 | $12.2M | Drafted 2012 but not in draft table; age-19 understated |
| Seth Curry | 2023 | 7->10 | 0.30->0.35 | $4.0M | Drafted 2013 but not in draft table; debuted 2013-14 |

Neither affects `is_max_contract` (both are well below their ceiling).

## 5. C2 segment check

No segment's |bias| grows by more than $0.3M:

| Segment | Old bias | New bias | |Growth| |
|---------|----------|----------|----|
| Bird Rights | -$2.44M | -$2.42M | -$0.015M |
| Minimum | +$1.97M | +$1.98M | +$0.002M |
| Sign & Trade | -$4.52M | -$4.49M | -$0.025M |
| Cap Space | -$0.23M | -$0.23M | +$0.003M |

(All other segments move by < $0.01M.)

## 6. Phase-3 collateral note

The brief identifies Reaves 2026 as the second-largest collateral row in the
phase-3 route-mixture re-run (+$9.71M of push damage). With the corrected
labels, Reaves moves from `is_max_contract = False` to `True` (cap_pct = 0.25
at ceiling 0.25 = 100%, well above the 90% threshold). He exits the collateral
set entirely — collateral rows are non-max rows above the classifier threshold,
and Reaves is now a max row. The $9.71M push damage vanishes from the
collateral list.

Butler 2019 makes the same transition (85.7% of ceiling -> 100%) and also
exits any collateral computation, though the brief did not flag him as a
collateral row.

The route-mixture branch was NOT recomputed (out of scope per the brief). The
phase-3 re-run should be repeated on the corrected labels to confirm the
collateral list shrinks as expected.

## 7. Source and coverage

**Source**: BBRef player index pages (`/players/a/` through `/players/z/`),
parsed from the "From" column (`data-stat="year_min"`). This is the same data
source as `height.py`. The "From" year uses BBRef's end-of-season convention;
`debut_season = year_min - 1` converts to our start-year convention.

**Coverage**: 5,416 players total; 789 of 793 training-data players matched
(99.5%). Duplicate names (46 across NBA history, 2 in our window — Brandon
Williams and Johnny Davis) are resolved by keeping the most recent debut.

**Spot-check** (5 players verified against public career records):

| Player | Expected debut | Actual | Source |
|--------|---------------|--------|--------|
| Austin Reaves | 2021 | 2021 | Undrafted, signed LAL summer 2021 |
| Jimmy Butler | 2011 | 2011 | Drafted 2011 by CHI |
| LeBron James | 2003 | 2003 | Drafted 2003 by CLE |
| Giannis Antetokounmpo | 2013 | 2013 | Drafted 2013 by MIL |
| Nikola Jokic | 2015 | 2015 | Drafted 2014, sat one year |

Jokic is the key test: drafted 2014 but debuted 2015-16. The debut-based
source correctly gives service 2015, while draft-year would give 2014 and
age-19 would give approximately the same.

## 8. Files touched

| File | Change |
|------|--------|
| `src/scraping/debut.py` | **NEW** — scraper for BBRef player index debut seasons |
| `data/processed/debut_seasons.csv` | **NEW** — 5,416 players with debut_season |
| `src/model/train.py` | Added `_load_debut_seasons()`; rewired `_compute_max_eligible` to use debut seasons as primary source |
| `data/raw/raw_external/designated_ineligible.csv` | Added Butler 2019 (team-change entry) |
| `data/raw/html_cache/` | 26 new cached pages (`/players/{a-z}/`) |

## 9. Proposed commit message

```
Real service years for the tier ceilings (ISSUES #20)

Replace the age-19 fallback in _compute_max_eligible with per-player
debut seasons scraped from BBRef player index pages. Covers 99.5% of
training-data players; 1 evaluation-frame row remains on the fallback.

The fix corrects Reaves 2026 (ceiling 0.30 -> 0.25, is_max True) and
Butler 2019 (ceiling 0.35 -> 0.30, is_max True), growing the max zone
68 -> 70. Butler also added to designated_ineligible.csv (team change
PHI -> MIA voids DVE eligibility). 119 base-tier movements total (117
down, 2 up), zero over-ceiling rows, frame stays 944.

Paired ΔSel: +0.00001 (t = +0.08) — neutral as expected for a
correctness fix. Max zone MAE $4.44M -> $4.40M. No C2 segment
regresses by more than $0.3M.
```

## 10. ISSUES.md

ISSUES #20 is fixed by this change. The entry should be deleted upon
adoption. No new issues discovered.
