# RESULT — Player-row feature missingness repair

Date: 2026-07-26

## Decision

Land the source and missingness-semantics repair. It restores observed facts;
it does not invent RAPM or rate statistics for players who did not play. The
production evaluation frame remains **944 rows** and includes the previously
waived/bought-out cases. Waiver history is handled by the separate `is_waived`
feature; Kendrick Nunn 2020 remains the continuation defect in ISSUES #31.

The source repair landed as **v8.1x** (`29b6383`); the waiver feature followed
as **v8.2x** (`800bed7`). The fixed-row harness below was rerun after v8.2x and
holds `is_waived` in both arms, so it isolates missingness semantics rather than
mixing the two changes. No confirmation split was opened.

## Repair

- Basketball Reference advanced statistics now provide an eight-season
  workload and box-rate fallback. The join tries normalized name plus season,
  then stable `player_url` plus season, fills only absent values, and never
  synthesizes RAPM.
- Mid-season multi-team rows retain the `TOT` or `2TM`/`3TM` aggregate. The
  final 4,474-row table has no duplicate `(player_url, season)` keys.
- Availability uses the exact `s`, `s-1`, `s-2` window, skips unknown source
  seasons with weight renormalization, preserves explicit zero games, and
  exposes `availability_3yr_coverage`. The shortened 2020-21 season uses 72
  games.
- Forty-six full-table zero-game player-seasons now carry
  `games=minutes=mpg=0` and `workload_source_status=did_not_play`.
- Stable NBA id and season coalesce impact-source name variants. Post-merge
  player history plus a sourced identity correction register closes the full
  table's age and height gaps.
- The advanced-stat refresher tries Playwright Chromium before the installed
  Edge channel. A failed season keeps stale rows; a failed uncached season
  refuses to overwrite the file with partial data.

## Missingness audit

`scripts/audit_feature_missingness.py` writes a row-level CSV and summary JSON
before production median fill.

| Field | Full, 3,113 rows | Evaluation, 944 rows | Interpretation |
|---|---:|---:|---|
| age | 0 | 0 | complete |
| height | 0 | 0 | complete |
| mpg | 0 | 0 | complete |
| availability | 0 | 0 | complete |
| usage / AST | 46 | 11 | all `did_not_play`; undefined rates |
| RAPM | 165 | 56 | source coverage; not synthesized |
| LEBRON | 63 | 16 | source coverage |
| `is_waived` | 819 | 152 | explicit Spotrac coverage status |

The current evaluation audit classifies 883 rows complete, 50 as partial
impact coverage, and 11 as did-not-play. There are no unresolved identity,
workload, or propagated-availability rows.

## Fixed-row evaluation

`scripts/eval_missingness_repair.py` reconstructs the old missingness semantics
in memory while holding the current 944 keys, 15-feature waiver model, target,
player folds, and ten seeds fixed.

| Metric | old | repaired |
|---|---:|---:|
| A1 pooled R2 | 0.8180 | **0.8214** |
| A1 pooled MAE | $2.816M | **$2.740M** |
| A1 selection R2 | 0.8111 | **0.8154** |
| A1 selection MAE | $2.894M | **$2.799M** |
| A2 R2 | **0.8647** | 0.8629 |
| A2 MAE | $3.020M | **$2.972M** |
| B1 R2 | 0.8493 | **0.8559** |
| B1 MAE | $3.104M | **$3.065M** |

Selection paired delta is **+0.0088 +/- 0.0080, t=1.10**. The old-missing
100-row slice improves from $3.94M to $3.00M MAE, and 2023 R2 improves from
0.789 to 0.843. The worst C2 absolute-bias growth is $0.203M in Cap Space,
inside the $0.30M guard.

The evidence is not a clean model-selection pass: A2 R2 falls 0.0018, even as
A2 MAE and bias improve, and calibration slope moves 0.978 to 0.971. Relative
calibration deterioration is about 0.0067, exceeding the 0.005 guard by about
0.0017. The repair is retained as data correctness, not presented as a proven
predictive-feature win. This result records that exception explicitly.

## Verification

- deterministic rebuild twice produced 3,113 rows and 40 columns;
- advanced fallback covers 2019-2026 with unique stable keys;
- full and evaluation immutable/workload fields have zero missing values;
- focused identity, availability, zero-game, waiver, and refresh-cache tests
  pass;
- compilation, audit generation, and whitespace checks pass.

Evaluation artifacts are
`outputs/models/missingness_repair_evaluation.json` and
`outputs/models/missingness_repair_oof.csv`; audit artifacts are under
`outputs/diagnostics/feature_missingness_*`.
