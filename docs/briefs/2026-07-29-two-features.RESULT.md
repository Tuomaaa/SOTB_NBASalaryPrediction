# Two floor-crash features: role mismatch and size for position

**Pin:** `470d389` (current `master` at dispatch)

**Branch:** `codex/two-features-20260729`

**Evaluation frame:** 944 rows, 803 selection / 141 confirmation, 10 seeds,
5-fold GroupKFold by player. All decision gates below exclude confirmation
players; pooled A1/A2/B1 are retained as the suite's reporting numbers.

## What changed and why

`scripts/build_external_features.py` now owns the two pre-registered ship-form
builders. Role mismatch is a fixed joint signal for high usage with inefficient,
low-playmaking offense; size for position is the season-position mean height
minus player height. Candidate NaNs remain native. Explicit missingness columns
exist only for the required coverage-control arms.

`scripts/eval_two_features.py` attaches those exact columns to the incumbent
frame and scores the incumbent, three coverage controls, `ts_plus`, the joint
composite, the composite plus its parts, and position-relative height. A feature
arm adds its columns to both the Grabit regression and route classifier, which
is the behavior an adopted `FEATURE_COL` would have on this pin.

## Registered forms and provenance

The role form was fixed before any score was run. Within each season of the
complete 3,880-row `impact_metrics.csv` population:

```text
u = max(z(usage_pct), 0)
e = max(-z(ts_plus), 0)
a = max(-z(ast_pct), 0)
r = max(-z(ast_tov_ratio), 0)
role_mismatch = u * mean(e, a, r)
```

All four ingredients must exist; otherwise the composite is native NaN. The
parts arm adds `u`, `e`, `a`, and `r` beside the composite. Evaluation-frame
coverage is 94.0% for `ts_plus` and 93.6% for the composite.

Position is not a function of the repo's `height_inches`. From the earliest
visible implementation (`f328639`), `src/scraping/advanced.py` passes
`position` straight through from nbarapm's `LAKER_history`; BBRef height is
merged separately. Across 3,387 player-seasons with both fields, one height can
carry all five positions, 99.1% of rows have a height shared by multiple
positions, and 260 players change position while their recorded height stays
fixed. Part 2 therefore continued. Its registered form is:

```text
undersized_for_position = mean(height_inches | season, position) - height_inches
```

Positive means undersized. Means use the full impact-metrics population, not
the evaluation frame. Four current-impact names collide with historical BBRef
names; the builder uses the project's existing "most recent identity" rule,
choosing the highest BBRef URL suffix. Missing position is native NaN, not a
synthetic position group. Final evaluation coverage is 87.8%.

## Headline numbers

These are the suite's pooled reporting metrics. `Delta Sel` is the decision-grade
paired fold delta against the incumbent.

| arm | A1 R2 | A2 R2 (2024-26) | B1 forward R2 | Delta Sel +/- SE | t | verdict |
|---|---:|---:|---:|---:|---:|---|
| incumbent | 0.798933 | 0.851546 | 0.835580 | - | - | keep |
| `ts_plus` | 0.798872 | 0.850979 | 0.833427 | -0.00086 +/- 0.00134 | -0.64 | do not adopt |
| role mismatch | 0.798854 | 0.850938 | 0.834919 | -0.00011 +/- 0.00131 | -0.08 | do not adopt |
| role + parts | 0.796506 | 0.844323 | 0.833789 | -0.00209 +/- 0.00290 | -0.72 | do not adopt |
| size for position | 0.797480 | 0.849021 | 0.835226 | -0.00185 +/- 0.00204 | -0.91 | do not adopt |

The old `ts_plus` rejection reproduces: its paired delta is negative, rather
than the recorded `< +0.001`. The composite improves on `ts_plus` by only
`+0.00075 +/- 0.00182` (`t=0.41`); the parts form is `-0.00123` worse than
`ts_plus` (`t=-0.48`). The interaction therefore does not beat the already
rejected main effect.

### Per-fold pairing

| arm | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---:|---:|---:|---:|---:|
| `ts_plus` | -0.00070 | +0.00070 | +0.00018 | +0.00153 | -0.00603 |
| role mismatch | +0.00073 | +0.00344 | +0.00088 | -0.00443 | -0.00118 |
| role + parts | -0.00917 | -0.00493 | +0.00498 | +0.00462 | -0.00596 |
| size for position | -0.00035 | -0.00956 | +0.00230 | -0.00180 | +0.00017 |

### Coverage controls

| control arm | A1 | A2 | B1 | Delta Sel +/- SE | t | candidate increment over control |
|---|---:|---:|---:|---:|---:|---:|
| `ts_plus_missing` | 0.798679 | 0.851209 | 0.835218 | -0.00039 +/- 0.00059 | -0.67 | `ts_plus` -0.00047 (t=-0.55) |
| `role_mismatch_missing` | 0.798790 | 0.851286 | 0.835327 | -0.00028 +/- 0.00065 | -0.43 | composite +0.00017 (t=0.12) |
| same role control | 0.798790 | 0.851286 | 0.835327 | -0.00028 +/- 0.00065 | -0.43 | role + parts -0.00181 (t=-0.70) |
| `undersized_for_position_missing` | 0.798975 | 0.851049 | 0.835173 | -0.00014 +/- 0.00108 | -0.13 | size -0.00170 (t=-0.70) |

No candidate gain is carried by missingness. The role composite is nominally
above its control, but the increment is only +0.00017 with `t=0.12`.

## Gate verdicts

The bars are the brief's: Delta Sel >= +0.002 and `t>2`; A2 same direction;
selection-only C2 worst |bias| growth <= $0.30M; selection-only B1 drop <=
0.003; relative C1 excess <= 0.005; and positive increment over coverage.

| arm | paired | A2 delta | C2 worst growth | B1 drop | C1 excess | coverage | final |
|---|---|---:|---:|---:|---:|---|---|
| `ts_plus` | **fail** | -0.00124 fail | +$0.055M pass | +0.00253 pass | -0.00195 pass | fail | **do not adopt** |
| role mismatch | **fail** | -0.00085 fail | +$0.030M pass | +0.00074 pass | -0.00107 pass | pass | **do not adopt** |
| role + parts | **fail** | -0.00806 fail | +$0.138M pass | +0.00205 pass | +0.00129 pass | fail | **do not adopt** |
| size for position | **fail** | -0.00295 fail | +$0.044M pass | +0.00031 pass | +0.00098 pass | fail | **do not adopt** |

All four fail gate 1 and the A2-direction guard. None reaches a later gate in
the formal sequence; later columns are still reported to expose collateral.

## Motivating rows

Values are signed OOF dollar errors (`prediction - observed`); every value is
an overprediction. Thus a smaller positive number is better.

| row | incumbent | `ts_plus` | role | role + parts | size |
|---|---:|---:|---:|---:|---:|
| Kelly Oubre Jr. 2023 | +$17.994M | +$18.003M | +$18.092M | +$18.592M | +$18.011M |
| Andre Drummond 2021 | +$16.290M | +$15.264M | +$15.689M | +$15.378M | +$16.149M |
| Emmanuel Mudiay 2019 | +$8.069M | +$8.271M | +$8.191M | +$8.157M | +$8.372M |
| Montrezl Harrell 2022 | +$17.141M | +$17.277M | +$16.641M | +$16.898M | +$16.391M |
| Langston Galloway 2020 | +$5.415M | +$5.491M | +$5.347M | +$5.360M | +$5.162M |

The role signal helps Drummond but worsens Oubre and Mudiay; it does not behave
like the hypothesized general three-player mechanism. Size helps both named size
cases (Harrell by $0.75M, Galloway by $0.25M), but the fixed pooled gates do not
move with those two cases.

## Floor and crash segments

Each cell is `MAE / bias` in millions on fixed rows. The five known bad-input
rows (Harrell 2023, Cousins 2020, Bol Bol 2023, Beasley 2023, Beverley 2023)
remain in the required 34-row report but are excluded from the 29-row
interpretation column.

| arm | Floor Zone (n=240) | crash 34 | valid-input crash 29 |
|---|---:|---:|---:|
| incumbent | 2.100 / +2.036 | 8.682 / +8.682 | 9.153 / +9.153 |
| `ts_plus` | 2.117 / +2.054 | 8.727 / +8.727 | 9.141 / +9.141 |
| role mismatch | 2.103 / +2.038 | 8.690 / +8.690 | 9.124 / +9.124 |
| role + parts | 2.113 / +2.049 | 8.719 / +8.719 | 9.121 / +9.121 |
| size for position | 2.103 / +2.037 | 8.684 / +8.684 | 9.109 / +9.109 |

Size reduces valid-input crash MAE by $0.044M while leaving the full Floor Zone
effectively unchanged (+$0.003M MAE). This is a real targeted observation, but
far smaller than the two named cases and opposite the pooled/A2 direction.

## Standard C2 table

Pooled fixed-row mechanism biases in $M are reported here; the gate itself uses
the selection-only rows above.

| mechanism (n) | incumbent | `ts_plus` | role | role + parts | size |
|---|---:|---:|---:|---:|---:|
| Bird Rights (274) | -2.33 | -2.37 | -2.37 | -2.38 | -2.35 |
| Cap Space (76) | -0.24 | -0.27 | -0.24 | -0.33 | -0.28 |
| Early Bird (46) | -1.73 | -1.72 | -1.73 | -1.75 | -1.77 |
| MLE (123) | +0.58 | +0.61 | +0.56 | +0.57 | +0.60 |
| Minimum (242) | +1.99 | +2.00 | +1.99 | +2.00 | +1.99 |
| Non-Bird (20) | +2.07 | +2.12 | +2.08 | +2.11 | +2.04 |
| Other (18) | +0.95 | +0.92 | +0.93 | +0.91 | +0.94 |
| Sign & Trade (13) | -4.52 | -4.49 | -4.50 | -4.47 | -4.53 |
| Unknown (131) | +0.38 | +0.38 | +0.39 | +0.37 | +0.39 |

## Ship-form statement and anomalies

- The evaluator imports `build_floor_crash_features` directly from
  `build_external_features`; there is no experimental reimplementation. The
  production merge and evaluation merge therefore use the same table and native
  NaNs byte-for-byte.
- Candidate derivation uses the full external impact population and no salary,
  target, confirmation flag, or evaluation-frame fill statistic. Decision
  metrics exclude all 141 confirmation rows.
- The first size run incorrectly grouped null positions together. That run was
  discarded before interpretation; the final 87.8%-coverage run leaves those
  rows NaN and has separately versioned checkpoints.
- No Delta Sel exceeds +0.01 (or is positive at all), so the re-verification
  trigger did not fire. The incumbent reproduces the brief's Floor Zone exactly:
  n=240, MAE $2.10M, bias +$2.04M.
- The filter prints 346 continuation demotions (334 dated, 12 consensus), one
  fewer than ISSUES #2's old category-level check, while the load-bearing final
  frame remains exactly 944. This is a reporting-category shift after later
  ceiling work, not a row-set change in this task.

## Files and artifacts

- `scripts/build_external_features.py` — canonical candidate builders and merge
- `scripts/eval_two_features.py` — reproducible evaluation harness
- `outputs/models/two_features/evaluation.json` — complete metrics, fold x seed
  matrices, gates, segment tables, and row evidence
- `outputs/diagnostics/two_features_oof.csv` — row-aligned OOF/forward predictions
- `docs/briefs/2026-07-29-two-features.RESULT.md` — this evidence bundle

No `ISSUES.md` addition: the only implementation defect found (null position as
a category) was fixed before the final run; the five upstream bad rows were
already filed by the parallel repair work and were not changed here.

**Proposed commit message:**
`Evaluate role-mismatch and position-relative size features`
