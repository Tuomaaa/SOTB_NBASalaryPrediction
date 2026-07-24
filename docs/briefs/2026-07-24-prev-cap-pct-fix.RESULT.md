# RESULT: real prior pay — 2018 salary scrape + prev_cap_pct repair

Covers ISSUES #13 and the rookie-exit anchor question. Branch
`worker/prev-cap-pct-fix`, pinned to `ddf4bae` (v7.9x).

## 1. What changed and why

`prev_cap_pct`, the model's second-strongest anchor, was fabricated for a large
minority of rows. The feature builder (`scripts/phase3.build_contract_features`)
derives it by looking back inside `training_data_v2.csv` (2019+) only, so any
fresh signing whose previous contract began before 2019 — **all** of season 2019,
plus later vets coming off long pre-2019 deals — took the rookie-scale slot fill
instead of its real prior pay (Klay 2019: 0.037 pick-11 slot money in place of
his true 0.186). Part 1 scrapes the 2018-19 team salary tables so that prior is
observable; Part 2 has the builder consult it.

**Part-2 root cause (audited before patching).** Klay's 2020-2026 rows also read
0.037, and the brief flagged two suspects — a bad `year_in_contract` in
`contract_structure_v2.csv`, or the year-1 lookback skipping a filtered prior
year-1 row. **Neither.** His structure table is correct (year_in_contract
1,2,3,4,5,1,2,3). The real cause is plainer: the `prev_cap_pct` loop assigns a
value **only to `year_in_contract==1` rows** (each gets the previous in-table
year-1 row's `cap_pct`). Klay 2019 is his first in-table year-1 (his prior
year-1 was 2014, pre-table) → never assigned → slot fill; his escalator rows
2020-23 / 2025-26 have `year_in_contract≥2`, so the loop never touches them →
slot fill. Klay 2024 (a genuine second in-table year-1) is correctly 0.300. The
fix is therefore not a patch to the lookback but a **previous-season fallback**:
after the year-1 loop, fill every remaining row from observed previous-season
pay — the same union of the salary table and `salaries_prehistory.csv` that
`_load_prev_season_cap_pct` already anchors the ceiling rule on — leaving the
slot fill only for true first contracts. Klay 2019 → 0.186, escalators → their
real priors, 2024 stays 0.300.

## 2. Part 1 — scrape coverage

Source: BBRef team-season pages `/teams/<ABBR>/2019.html` (`salaries2` table),
season label **2018** (= 2018-19). Fetched via the sanctioned cached,
rate-limited path (a reused-browser cache-warm pass was used because a fresh
browser per page 403'd ~half of a burst; all 30 pages ultimately cached, BRK
needed a second pass).

- **30/30 teams parsed.** 538 distinct 2018 players scraped.
- **EXTEND-ONLY honoured**: 153 pre-existing 2018 rows kept byte-for-byte, **399
  new players added**, **0 conflicts** (every overlap agreed within $50k — the
  team-page season-total convention matches the player-page values already
  there). `salaries_prehistory.csv`: 368 → 767 rows, 2018 players 153 → 552.
- **Eval-frame coverage**: season-2019 rows with an observed 2018 prior go from
  **41/147 (28%) → 136/142 (96%)**. The 6 still missing are two-way / minimum
  players whose salary is absent from the team salary table (Caruso, Milton,
  Iwundu, Gabriel, Amile Jefferson) plus one rename — **Enes Kanter → Enes
  Freedom** (his 2018 pay is on the page under the old name). All six carry
  negligible prior pay and fall back to slot fill. See ISSUES addition.

## 3. Prehistory side effects on the FRAME (reported, not folded into the delta)

Extending `salaries_prehistory.csv` feeds `_load_prev_season_cap_pct`, which also
drives the no-decrease ceiling floor and the continuation filter's escalator-step
signal. Measured (original vs extended prehistory, identical code):

- **Ceilings unchanged**: `max_eligible_pct` / `is_max_contract` /
  `tier_ceiling_pct` move on **0** rows.
- **Frame 949 → 944**: 5 season-2019 rows newly demoted by `_filter_continuations`
  — **Paul Millsap, Tony Snell, Langston Galloway, Jahlil Okafor, Yogi Ferrell**.
  Each is a genuine multi-year-deal escalator continuation kept as year-1 *only*
  because its 2018 prior was unobservable; ISSUES #2 names "Millsap 2019" as *the
  canonical kept suspect* and states expanding prehistory "converts more of them
  into decidable rows." This is that documented effect. None sit in the 2019
  FA-signings list (the veto); 0 un-demotions.

> **Row-count flag for the architect.** This is a filter-chain row-count change
> (949→944). It is fully explained and anticipated by ISSUES #2, and the decision
> metric below is *pinned to the 949 reference frame* so it is not confounded —
> but the landing frame is 944, and ratifying that is the architect's call.

## 4. Numbers — the arms

Decision metric: paired ΔSel on `fold_r2_selection`, **10 seeds × 5 GroupKFold
folds**, Grabit champion fitter, **frame pinned to the 949 reference world**, only
the `prev_cap_pct` column swapped. Ship-form guaranteed: the in-memory Arm A
column equals `phase3.build_contract_features` byte-for-byte on all 3,113 rows,
so the measured delta *is* the shipped delta. Reference pooled bias −$0.23M.

| arm | A1 | A2 | B1 | C1 slope | MAE $M | ΔSel ± SE | t |
|-----|------|------|------|------|------|------|------|
| **ref** (buggy) | 0.7631 | 0.8160 | 0.8191 | 0.988 | 3.29 | — | — |
| **A** repair-in-place | 0.7732 | 0.8232 | 0.8255 | 0.991 | 3.22 | **+0.0135** ± 0.0018 | **+7.60** |
| **B** redefine prev-season | 0.7738 | 0.8245 | 0.8286 | 0.988 | 3.20 | **+0.0155** ± 0.0017 | **+9.06** |

Per-fold ΔSel — A: `[.0135 .0187 .0118 .0156 .0080]`; B: `[.0152 .0199 .0136 .0185 .0104]` (every fold positive).
Head-to-head **B − A = +0.0020 (t=+7.00)**: B is small-but-consistently better.

**Motivating fixed-row segments** (bias $M, pred − actual):

| segment | n | ref | A | B |
|---|---|---|---|---|
| breakout (age≤24 & ≥$15M) | 69 | −3.04 | −3.18 | −3.16 |
| prime (25-29) | 434 | −0.48 | −0.46 | −0.45 |
| **season 2019** | 147 | **−0.74** | **−0.42** | **−0.43** |

The repair fixes the 2019 boundary (−0.74→−0.42). Breakout bias *worsens*
slightly (−3.04→−3.18): the now-reliable anchor is trusted more, and for
rookie-exit rows the anchor is still low slot money — this is the Part-3 motivation.

**C2 mechanism bias, worst |bias| growth vs ref** (fixed rows): A **+0.16**
(Early Bird), B **+0.19** (Early Bird) — both well under the $0.3M gate;
Sign & Trade and Non-Bird *improve* by 0.3-0.4.

### Part 3 — rookie-exit anchor (on the winning base; tested on both A and B)

`is_rookie_exit` = year-1 rows whose *previous* season was a rookie-scale season
(ex ante, from draft year/pick via `_load_rookie_scale_set`). **It describes the
PRIOR contract, not the current one** — the current row is a fresh market deal —
so it is *not* v4.0's leaked `is_rookie_scale`, which flagged the CURRENT contract
as rookie-scale (a proxy for the target being slotted money). 144 rookie-exit
rows in-frame. Under Arm A/B these rows already carry their real prior-season
(rookie) pay, so (i) status-quo = A/B above.

Increments shown over **base A** (base-B is materially identical; noted where it
differs):

| arm | A1 | A2 | B1 | ΔSel/t | breakout bias | worst C2 growth |
|-----|------|------|------|------|------|------|
| A (base) | 0.7732 | 0.8232 | 0.8255 | +0.0135/7.6 | −3.18 | +0.16 |
| (ii) + flag | 0.7772 | **0.8207↓** | 0.8245 | +0.0186/6.1 | −2.81 | **+0.35** (Early Bird) |
| (iii) NaN (native) | 0.7727 | 0.8231 | 0.8257 | +0.0129/8.1 | −3.18 | +0.16 |
| (iv) flag + NaN | 0.7762 | **0.8201↓** | 0.8244 | +0.0174/5.9 | −2.82 | +0.35 |
| (v) slot-average | **0.7821** | 0.8250 | 0.8254 | **+0.0243/8.4** | **−2.73** | **+0.42** (Other) |
| (v) + flag | 0.7822 | 0.8237 | 0.8240 | +0.0245/7.5 | −2.59 | +0.42 |

(v) slot-average is fold-honest (own row excluded, binned picks 1-3/4-10/11-20/
21-30/2nd+undrafted). Base-B mirrors this: (v) A1 0.7806, ΔSel +0.0241/6.71,
worst C2 **+0.41** (Other); flags again push A2 the wrong way.

## 5. Gate verdicts

**Part 2 (both arms) — PASS all four gates.**
1. ΔSel ≥ +0.002, t>2: A +0.0135 t=7.6, B +0.0155 t=9.06. ✔
2. Source control: the brief pre-cleared season-dummy (not a pure offset — the
   corrected values carry *within-season* signal: Klay 0.186 vs a min player's
   0.014 in the same 2019). The B1 forward veto *rising* (below) is the proof it
   is content, not a training-side season absorption. ✔
3. B1 forward veto moves WITH the gain: 0.8191 → 0.8255 (A) / 0.8286 (B). ✔
4. C2: worst mechanism |bias| growth +0.16 (A) / +0.19 (B) ≤ $0.3M. ✔

**Part 2 correctness-fix rule** (adopt unless ΔSel significantly negative): ΔSel
is significantly *positive*. **Adopt.**

**Part 3 — NO arm passes.**
- (ii)/(iv) flag: **fail gate 3** — A2 drops (0.8232→0.8207/0.8201) while A1
  rises, the donated-capacity / training-side-absorption pattern the forward veto
  exists to catch — **and** gate 4 (Early Bird +0.35 > $0.3M).
- (iii) NaN: neutral (ΔSel increment over A ≈ −0.0006); nothing to adopt.
- (v)/(v)+flag slot-average: best headline and it does lift the breakout class
  (−3.18→−2.73), but **fails gate 4** — Early Bird +0.37 and Other +0.42 (n=18)
  grow past $0.3M on the A base, Other +0.41 on the B base. `draft_pick` is
  already a feature, so the tree builds most of this itself; the explicit slot
  base-rate over-corrects segments it already handles. This is exactly the
  "donated capacity may well lose" caution in the brief, and the fold-honest
  P(mechanism|x) precedent (−0.0073) is the family it belongs to.

## 6. Anomalies

- **ΔSel above +0.01 (the red-flag line).** A +0.0135, B +0.0155 both exceed it.
  Explained, not spurious: this repairs the model's #2 feature on the largest
  season's rows (2019 prior coverage 28%→96%) plus every escalator. Localised to
  the 2019 segment (bias −0.74→−0.42), corroborated by the B1 forward lift, and
  free of leakage (prior-season pay is strictly ex ante). Not a red flag once the
  mechanism is named.
- **(v) slot-average ΔSel +0.0243** is even larger; it is fold-honest (own row
  excluded) so not target leak, but it is rejected on C2 regardless.
- **6 uncovered 2019 rows** incl. the Kanter→Freedom rename (ISSUES addition).

## 7. Files touched / branch

- `scripts/scrape_2018_salaries.py` (**new**) — 2018-19 team-salary scraper,
  extend-only with conflict reporting.
- `scripts/phase3.py` — previous-season fallback in `build_contract_features`
  (Arm A ship form) + import of `_load_prev_season_cap_pct`.
- `data/processed/salaries_prehistory.csv` — extended 368 → 767 rows (0 existing
  rows moved).
- `data/processed/training_data_v2.csv` — **only `prev_cap_pct` changed** (1,555
  rows; every other column byte-identical, verified cell-by-cell). Produced by
  applying stage 3 (`build_contract_features`) in isolation: the full
  `rebuild_training_data.py` chain validated clean (all 15 stable features 100%,
  cap_pct 100%) but *also* churns the cosmetic `team`/`agent`/`position` columns
  via ISSUES #5's known non-determinism, so stage 3 was applied alone to keep the
  diff to `prev_cap_pct` only.
- Branch `worker/prev-cap-pct-fix`. `outputs/models/evaluation_suite.json` NOT
  regenerated (gitignored; the architect regenerates the canonical suite on
  landing — this bundle's paired deltas are cleaner for the A/B decision anyway).

## 8. Recommendation

1. **Adopt Part 1** (scrape) and **Part 2 (the repair)** — a clean correctness
   fix, +0.0135–0.0155 ΔSel with every gate passing and the 2019 boundary bias
   halved.
2. **Ship Arm A** as the default (minimal change, preserves the "previous
   contract's year-1" semantics; it is what this branch ships). **Arm B** is a
   validated, simpler (uniform "previous-season actual", no year-1 loop),
   strictly-dominant alternative (+0.0020 t=7 over A, better A2/B1/MAE) — a
   one-function swap if the architect prefers it. Either is defensible.
3. **Adopt no Part 3 arm.** The rookie-exit breakout underpricing (−3.04, and it
   is the *same population* as task #4's max-classifier blind spot) is real, but
   marking the anchor fails the forward veto and relocating it fails C2. This is
   the max-classifier's job, not the anchor's.
4. **Ratify the 949→944 frame change** (5 documented continuation demotions).

## 9. Proposed commit message (architect edits / assigns the version)

```
Repair prev_cap_pct: real prior pay from a 2018 salary scrape (ISSUES #13)

build_contract_features derived prev_cap_pct only from the 2019+ table, so
every fresh signing coming off a pre-2019 deal — all of season 2019 and later
vets — took the rookie-slot fill instead of its real prior (Klay 2019: 0.037
for a true 0.186). Root cause is the year-1-only lookback, which also never
touches escalator rows; fix is a previous-season fallback from the salary
table + salaries_prehistory (the source the ceiling rule already uses).

Part 1 scrapes the 30 BBRef 2018-19 team salary tables (extend-only, +399
players, 0 conflicts); 2019 prior coverage 28%→96%. Part 2 repair: paired
ΔSel +0.0135 (t=7.6), 2019 segment bias −0.74→−0.42, B1 forward 0.819→0.826,
C2 max growth +0.16M. Only prev_cap_pct moves in training_data_v2.csv.

Extended prehistory also makes 5 stale 2019 continuations decidable
(Millsap et al.), frame 949→944 — ISSUES #2's documented decidability gain.
Rookie-exit anchor arms (flag / NaN / slot-average) all fail a gate; not adopted.
```

## 10. ISSUES.md

- **Delete #13** (fixed and verified: Klay 2019 → 0.186, no bogus fill where a
  real prior exists).
- **Add** a low-severity entry: `salaries_prehistory.csv` misses ~6 of the 2019
  eval rows — two-way/minimum players absent from the team salary table, plus the
  **Enes Kanter → Enes Freedom** rename (2018 pay on the page under the old name;
  a name-alias map would recover it).
