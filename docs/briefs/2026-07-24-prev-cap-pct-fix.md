# Task brief: real prior pay — 2018 salary scrape + prev_cap_pct repair

Covers ISSUES #13 and the rookie-exit anchor question. Read
`docs/worker-brief.md` first — isolation section, own `git worktree`,
`OMP_NUM_THREADS=6`.

**Pin**: branch off current `master` (`v7.9x`) in your own worktree. Deliver a
branch + `docs/briefs/2026-07-24-prev-cap-pct-fix.RESULT.md`.

## Why

`prev_cap_pct` is the model's second-strongest anchor (mpg alone 0.569 →
mpg+prev 0.607) and it is broken two ways:

1. **Every 2019 row carries fabricated prior pay.** `build_contract_features`
   (scripts/phase3.py) looks back only inside `training_data_v2.csv` (2019+),
   so any player whose previous contract began before 2019 — which is ALL 147
   rows of season 2019, plus later signings coming off long pre-2019 deals —
   gets the draft-slot rookie-scale fill. Klay Thompson 2019: true prior pay
   0.186 of the cap, feature says 0.037 (pick-11 slot money). The salary IS on
   record for 155 players in `salaries_prehistory.csv` and the filter helper
   `_load_prev_season_cap_pct` in train.py already reads it correctly; the
   feature builder just never looks.
2. **Rookie-exit rows anchor on administratively slotted money.** A player
   coming off a rookie-scale deal carries his slot's year-1 pay (Booker 0.034,
   MPJ 0.047) — a number the market ignores when pricing second contracts.
   Champion OOF evidence on the 949 frame: rows with age ≤ 24 and actual pay
   ≥ $15M (n=69) are biased **−$3.04M** (MAE $4.53M), while cheap young rows
   are nearly unbiased (−0.19) — the anchor only hurts the breakouts. This is
   the same population as the max-classifier blind spot in task #4.

## Part 1 — scrape the 2018-19 salary tables

Target: prior-season pay for every player active in 2018-19, so season-2019
rows can see real numbers.

- Source: BBRef team-season pages for the 2018-19 season
  (`/teams/<ABBR>/2019.html`, 30 pages), each carrying a salaries table.
  Live-fetch rules per CLAUDE.md: 3s+ delays, cache under
  `data/raw/html_cache/`, expect one team to 403 and need a retry. **A failed
  team page degrades to that team missing from the new rows — never to a
  deleted existing row.**
- Store into `salaries_prehistory.csv` with season label **2018** (= 2018-19,
  matching the existing Klay row). **EXTEND-ONLY**: existing rows byte-stable;
  if a scraped value conflicts with an existing row for the same
  (player, season), hard-fail and report — do not overwrite.
- Report: players added, total coverage of 2018, and how many of the 147
  season-2019 evaluation rows now have observable prior pay.

## Part 2 — repair `build_contract_features` (ISSUES #13)

Root-cause first: Klay's 2020-2026 rows ALSO read 0.037227 even though his
2019+ pay is in-table — find why the year-1 lookback misses them before
patching anything (suspect: his 2019 row's `year_in_contract` in
`contract_structure_v2.csv`, or the year-1-rows-only lookback skipping
players whose prior year-1 row was filtered). The audit result decides the
fix's shape; do not paper over it with a fallback.

Then two semantic arms, SHIP-FORM tested (the form that ships is the form
that is tested — median-fill vs native-NaN flipped a verdict once already):

- **Arm A (repair in place)**: keep the "previous contract's year-1 pay"
  semantics; consult `salaries_prehistory.csv` (via the same lookup
  `_load_prev_season_cap_pct` uses) when the prior contract predates the
  table, using observed prior-season pay as the proxy for its year-1. Slot
  fill remains only for true first contracts (no NBA history).
- **Arm B (redefine)**: `prev_cap_pct` = the player's PREVIOUS SEASON's
  actual cap_pct, everyone, from the season table + prehistory. Simpler, no
  proxying, fixes long-prior-deal vets that Arm A cannot reach (a 2021 signee
  whose prior deal ran 2017-2020 has its year-1 outside every table, but his
  2020 pay is in-table). This is a semantic change to a top feature — full
  challenger gates.

## Part 3 — rookie-exit encoding

On top of the winning arm, test the anchor question:

- (i) status quo (slot money as prev),
- (ii) add `is_rookie_exit` flag (prior contract was rookie-scale — derivable
  ex ante from draft year/pick + contract structure; this is NOT v4.0's
  leaked `is_rookie_scale`, which described the CURRENT contract — state the
  distinction in the RESULT so the ablation graveyard is not misread),
- (iii) `prev_cap_pct = NaN` for rookie-exit rows, native-NaN handling,
- (iv) flag + NaN.

## Judging

- **Part 2 bug repair (Arm A)** is a correctness fix: adopt unless the paired
  selection delta is significantly NEGATIVE (t < −2) or C2 breaks; report the
  delta either way. Precedent: v7.3x's fill improvement (+0.0034, t=4.91).
- **Arm B and Part 3 arms** are challenger changes: paired ΔSel t > 2, A2
  same direction, C2 segment |bias| growth ≤ $0.3M, B1 no drop > 0.003.
  Not season-aligned, so no season-dummy arm needed.
- **Motivating segments, reported for every arm** (fixed rows): the breakout
  class (age ≤ 24 & actual ≥ $15M: bias −3.04 → ?), prime vets (age 25-29:
  −0.48 → ?), season-2019 rows, and the C2 mechanism table.
- Rebuild via `scripts/rebuild_training_data.py` and verify **only the
  `prev_cap_pct` column changes** in `training_data_v2.csv` (count the rows);
  every other feature byte-stable.
- 10 seeds, 5 GroupKFold folds, per-fold deltas in the RESULT. If adopted,
  the landing is the next version tag — the architect's.

## Traps

- `salaries_prehistory.csv` is extend-only; `contract_structure_v2.csv` is
  untouchable (extend_contract_structure.py territory — if the Part-2 audit
  finds bad `year_in_contract` rows there, REPORT them, do not edit the
  table).
- BBRef 403s: retry the team once after a delay; a team that stays down is
  reported missing, not silently absent.
- Slot-fill values changing wholesale will move `prev_cap_pct` on hundreds of
  rows — that is the point, but the C2 table is what catches an unintended
  segment regression. Deltas above +0.01 are a red flag to re-verify.
- Do not touch FEATURE_COLS membership beyond Part 3's arms; wingspan etc.
  are parked elsewhere.

## Deliverable

Branch + RESULT per the evidence-bundle format: scrape coverage numbers, the
Part-2 root-cause paragraph, per-arm paired tables with the motivating
segments, ship-form statement, and a recommendation. Expected effort: a day.
