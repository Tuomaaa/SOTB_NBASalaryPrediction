# Plan — player-row feature missingness repair

Date: 2026-07-25

## Objective

Audit and repair feature missingness before model-level median fill. The first
pass covers both populations that matter:

- all **3,113** rows in `training_data_v2.csv`, including continuation and
  rookie-scale rows used for public Contract Surplus scoring;
- the final **936** Year-1 market-price rows used by the evaluation suite.

This is a data-integrity plan. It does not propose adding a predictive feature
until source recovery and missingness semantics are correct.

## Severity: high

The final evaluation frame has no missing age, height, prior pay, DARKO, draft
position or CBA-era values. The remaining problem is nevertheless systematic:

| population / field | missing | rate |
|---|---:|---:|
| Evaluation rows with any core missing value | 104 / 936 | 11.1% |
| Evaluation `availability_3yr` | 90 / 936 | 9.6% |
| Evaluation RAPM / usage / AST block | 56 / 936 | 6.0% |
| Evaluation `mpg` | 40 / 936 | 4.3% |
| Evaluation LEBRON | 16 / 936 | 1.7% |
| Full-table `availability_3yr` | 315 / 3,113 | 10.1% |
| Full-table RAPM / usage / AST block | 165 / 3,113 | 5.3% |
| Full-table `mpg` | 149 / 3,113 | 4.8% |
| Full-table age | 19 / 3,113 | 0.6% |
| Full-table height | 2 / 3,113 | 0.06% |

The missingness is not spread evenly. In the 2023 evaluation season,
RAPM/usage/AST are missing on **24.0%** of rows, mpg on **18.4%**, and
availability on **19.2%**. This is a source-coverage event, not random player
noise.

Current OOF residuals reinforce the priority but do not prove causality:

| group | n | MAE | bias |
|---|---:|---:|---:|
| Complete core features | 832 | $2.685M | +$0.057M |
| Any core value missing | 104 | $4.065M | -$0.851M |
| LAKER block missing | 56 | **$4.746M** | **-$1.199M** |
| Availability only missing | 43 | $3.387M | -$0.722M |

The LAKER-block population contains large contracts and both error directions;
it is not just low-salary fringe noise. Jaren Jackson Jr. 2026, Kyle Kuzma
2023, Jordan Clarkson 2023, De'Andre Hunter 2023, Keldon Johnson 2023 and
Dillon Brooks 2023 each carry more than $11M of current absolute error.

## Diagnosis

### 1. LAKER regular-season coverage is the main defect

Forty evaluation rows have no mpg and are also missing RAPM, usage and AST.
Another 16 have repaired workload or partial source coverage but still lack the
RAPM/usage/AST block. The 40 rows with missing workload are:

| season | player rows |
|---:|---|
| 2019 | Wenyen Gabriel |
| 2020 | Jontay Porter |
| 2021 | Jonathan Isaac; Omer Yurtseven; Zach Collins |
| 2022 | Jason Preston; T.J. Warren |
| 2023 | Ayo Dosunmu; De'Andre Hunter; Dillon Brooks; Drew Eubanks; E.J. Liddell; Grant Williams; Jevon Carter; Jock Landale; Jordan Clarkson; Keldon Johnson; Kevin Love; Kevin Porter Jr.; Kyle Kuzma; Naz Reid; Nick Richards; Omer Yurtseven; P.J. Washington; Seth Curry; Torrey Craig; Trey Lyles; Tristan Thompson; Ty Jerome; Yuta Watanabe |
| 2024 | Jarred Vanderbilt; Kelly Oubre Jr.; Luka Garza |
| 2025 | Bobi Klintman; N'Faly Dante; Tim Hardaway Jr. |
| 2026 | Gary Trent Jr.; Jabari Smith Jr.; Tim Hardaway Jr.; Trey Lyles |

This population mixes three cases that must not share one fallback:

1. active players omitted by a source-year or parser failure;
2. a valid name/NBA-id row split;
3. players who genuinely logged zero regular-season games, where games and mpg
   should be observed zeros while rate statistics remain unavailable.

### 2. Availability has a propagation bug

`compute_availability` currently sums the current and prior `gp_pct` values
without excluding missing observations. One missing season therefore makes
later three-year windows NaN. Of the 90 evaluation rows missing availability:

- 40 also lack current workload;
- **50 have current workload but inherit NaN from another season** (43
  availability-only rows plus seven partial LAKER rows).

This is the cleanest first repair: a missing source row should not silently
erase known games from later seasons. A genuine missed season must first be
recorded as `games=0`, so skipping unknown observations cannot turn an injury
absence into apparent availability.

### 3. All-row identity still matters outside training

The filtered evaluation frame has zero missing age and height, but the complete
3,113-row scoring table has 19 missing ages and two missing heights. Most are
partial-season or filtered rows, but six full-salary Tim Hardaway Jr. rows
(2019-2024) and two Jabari Smith Jr. rookie-scale rows are included. They do not
affect Signing Residual evaluation, but the web export scores all rows for
Contract Surplus and currently substitutes training medians.

The existing age repair runs on impact history before the salary merge. It
should be followed by a post-merge player-history repair so age available from
the salary side or another merged season can fill the complete scoring table.

## Work plan

### Phase 0 — make missingness observable (no model-number change)

1. Add `scripts/audit_feature_missingness.py` with two named frames: full
   scoring rows and final evaluation rows before `_prepare_Xy` imputation.
2. Write a row-level CSV containing player, season, filter status, missing
   fields, missing-count, and a source-status classification.
3. Write a summary JSON with counts by feature, season and missingness pattern.
4. Add hard checks for immutable or directly observed fields. Age, height,
   games and mpg may not reach model median fill without an explicit
   `source_unavailable` or `did_not_play` reason.

### Phase 1 — repair availability propagation

1. Classify every missing `games` row before changing the rolling calculation:
   observed zero, source omission, or unresolved.
2. Change `compute_availability` to retain chronology but renormalize weights
   over observed seasons. An explicitly observed zero remains in the weighted
   average; an unknown source observation is skipped and reduces coverage.
3. Produce `availability_3yr_coverage` for auditing. Do not add it to
   `FEATURE_COLS` without a separate ablation.
4. Add tests for missing current, missing prior, missing middle, explicit zero,
   and a fully observed three-season window.

Expected immediate result: the 50 propagated NaNs disappear. Remaining
availability gaps must correspond one-for-one with unresolved current workload.

### Phase 2 — recover the LAKER block

1. Audit the 2023 source response first. Its 23 missing-workload evaluation
   rows establish a systematic season-level failure and should be repaired in
   the scraper/parser, not as 23 permanent manual exceptions.
2. Re-fetch or reparse regular-season LAKER rows keyed by stable NBA id plus
   season. Keep the existing name coalescing only as identity reconciliation.
3. For genuine zero-game seasons, record `games=0`, `minutes=0`, `mpg=0` with a
   source. Leave RAPM/usage/AST missing because those rates are undefined.
4. Use `impact_metric_corrections.csv` only for narrow source omissions that
   survive the systematic repair. Each value needs its own public source.
5. Re-run the full 3,113-row audit, not only the 40 evaluation rows; otherwise
   continuation rows can remain wrong in web output.

### Phase 3 — close all-row identity gaps

1. Run age inference again after the salary-impact merge, using a unique
   player-specific `age - season` offset. Fail on conflicting offsets.
2. Repair the two height rows through the existing height source or a narrow
   sourced correction.
3. Assert zero missing age and height in both the full scoring table and final
   evaluation frame.

### Phase 4 — decide partial impact missingness

After source recovery, audit the remaining LEBRON/RAPM/usage/AST gaps:

- Undefined for a zero-game season: retain NaN with an explicit reason.
- Missing despite a played season: repair the source or exclude the metric from
  that row's evidence; do not silently describe it as an average player.
- If production still median-fills tree inputs, compare median fill with native
  NaN and include a pure missingness/season control. Missingness is strongly
  season-aligned, so any apparent gain must beat the project's season-dummy
  guard.

## Evaluation protocol

This work changes values and may change no rows, but it still moves model
numbers. Use the same version discipline as the floor-crash repair:

1. Freeze the current 936 evaluation keys and player-grouped folds.
2. Fit old and repaired data states on the same validation players over 10
   seeds; report A1 selection/pooled, A2 and MAE with per-fold paired deltas.
3. Report 2023 separately because it contains the source outage.
4. Run C2 mechanism bias, Floor/Max zone MAE and the confirmation canary.
5. Run the native-NaN and season-dummy controls if imputation semantics change.
6. Run the ordinary suite for the new standalone A1/A2/B1, but do not compare
   standalone R2 if the row set changes.

## Acceptance criteria

- Full and evaluation frames: zero unreasoned age, height, games or mpg fills.
- Evaluation frame: zero propagated availability NaNs; any remaining NaN has an
  explicit unresolved-source reason and coverage value.
- The 2023 played-season LAKER block is recovered systematically.
- True zero-game seasons carry observed zero workload, not median workload.
- Remaining rate-stat NaNs are documented as undefined or source-unavailable.
- Rebuild is deterministic and all focused tests pass.
- Fixed-row C2 absolute-bias growth is at most $0.3M; A2 moves in the same
  direction as the paired A1 result or the disagreement is explained.
- Any missingness-based model change beats its season-dummy control.
- Because numbers will move, the landing takes the next version and records
  A1/A2/B1 plus the before/after missingness table.

## Reproduction baseline

Run the production filter chain without calling `_prepare_Xy`, then count NaN
over `FEATURE_COLS`. Merge `outputs/models/oof_reference.csv` by
`(player_name_norm, season)` for the residual slices above. The baseline counts
that the implementation must reproduce before changing anything are:

```text
full rows: 3113
evaluation rows: 936
evaluation any core missing: 104
evaluation availability missing: 90
evaluation RAPM/usage/AST missing: 56
evaluation mpg missing: 40
full age missing: 19
full height missing: 2
```
