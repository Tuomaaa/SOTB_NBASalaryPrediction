# Methodology — NBA Free Agent Valuation Model

## Objective

Predict a player's market value as **cap_pct** (annual salary / salary cap), using prior-season performance data. Inspired by Hollinger's BORD$, but fully open-source and reproducible.

## Data Sources

| Source | Data | Rows |
|--------|------|------|
| Basketball Reference (`/teams/TEAM/YEAR.html`) | Per-season salary, age, team | 4,397 player-seasons (1,037 players, 2019–2030) |
| nbarapm.com (`/search/DARKO`, `/search/lebron`, `/search/LAKER_history`) | DARKO DPM, LEBRON, RAPM, BPM, usage, shooting splits, box-score rates | 3,880 player-seasons (903 players) |
| Basketball Reference (`/playoffs/NBA_YEAR_advanced.html`) | Playoff BPM per player-season | 1,361 player-seasons |
| Basketball Reference (player index pages) | Height in inches | 5,416 players |
| Basketball Reference (team season pages) | Team win-loss record | 209 team-seasons (30 teams × 7 seasons) |
| Manual reference | Salary cap by year, CBA parameters | Hand-curated |

## Season Convention & Intentional Lag

A critical design decision: salary data uses the **start year** (season=2025 → 2025-26 salary) while nbarapm.com uses the **end year** (season=2025 → 2024-25 stats). When merged on `season`, this creates a **1-year lag**: prior-season stats predict current-season salary. This is intentional — teams sign contracts based on prior performance.

## Target Variable

```
cap_pct = annual_salary / salary_cap_for_that_season
```

Range: 0.0 to ~0.35 (max contract). All dollar amounts are normalized to cap percentage to enable cross-era comparison.

## Training Data Construction

1. **Inner join** salaries × impact metrics on `(player_name_norm, season)` → 3,059 matched rows
2. **Merge** height data on normalized name → 3,057/3,059 matched
3. **Merge** team wins on `(team_abbreviation, season)` → 2,530/3,059 matched
4. **Contract structure detection**: salary progression analysis detects contract boundaries (new contract when salary ratio falls outside 0.90–1.15, team changes, or season gaps). Each row tagged with `year_in_contract` and `contract_years`.
5. **Year-1 filter**: only keep `year_in_contract == 1` rows. Years 2+ of any multi-year contract are CBA-mandated escalators (5% or 8% raises), not fresh market evaluations. This reduces noise where the model would otherwise learn to predict locked-in salary bumps as if they reflected current-season performance. Reduces dataset from 3,059 → **1,739 rows**.
6. **Rookie scale filter (Ridge only)**: remove rows where `age ≤ 23 AND cap_pct ≤ 0.10` (523 rows). Rookie scale salaries are slotted by draft position, not market-negotiated. XGBoost keeps these rows and uses `is_rookie_scale` as a feature instead.

Final training set: **1,216 player-seasons** (Ridge) / **1,739 player-seasons** (XGBoost).

## Feature Set (25 features)

### Performance Metrics (z-scored within season)
| Feature | Description | Source |
|---------|-------------|--------|
| `darko_dpm_z` | DARKO Daily Plus-Minus, z-scored by season | nbarapm.com |
| `lebron_z` | LEBRON metric, z-scored by season | nbarapm.com |
| `rapm_z` | Regularized Adjusted Plus-Minus, z-scored by season | nbarapm.com |

Z-scoring within season normalizes for era differences (e.g., 2020 COVID season vs normal 82-game seasons).

### Box Score & Shooting
| Feature | Description |
|---------|-------------|
| `bpm` | Box Plus-Minus (season-level) |
| `ts_plus` | True Shooting % relative to league average (100 = average) |
| `fg3_plus` | 3-point FG% relative to league average |
| `threepar_plus` | 3-point attempt rate relative to league average |
| `ast_pct` | Assist percentage |
| `ast_tov_ratio` | Assist % / Turnover % |

### Age & Experience
| Feature | Description |
|---------|-------------|
| `age` | Player age at time of season (from nbarapm.com) |
| `age_squared` | Captures nonlinear decline after peak |
| `experience_years` | Years in league |

### Workload & Availability
| Feature | Description |
|---------|-------------|
| `minutes` | Total regular-season minutes |
| `usage_pct` | Usage rate (offensive load) |
| `availability_3yr` | Weighted GP% over past 3 seasons (0.5/0.3/0.2) |

### Physical
| Feature | Description |
|---------|-------------|
| `height_inches` | Player height from BBRef |

### CBA Constraints
| Feature | Description |
|---------|-------------|
| `cba_era` | Binary: 0 = pre-2023 CBA, 1 = post-2023 CBA |
| `max_eligible_pct` | Max contract % player is eligible for (25/30/35% based on experience) |
| `is_vet_min` | Binary: cap_pct ≤ 0.025 (training only — see Known Issues) |
| `is_mle_range` | Binary: 0.025 < cap_pct ≤ 0.08 (training only) |
| `is_rookie_scale` | Binary: age ≤ 23 AND cap_pct ≤ 0.10 (training only) |

### Team & Playoff Context
| Feature | Description |
|---------|-------------|
| `win_pct` | Team regular-season win percentage |
| `made_playoffs` | Binary: did the player's team make the playoffs? |
| `playoff_bpm_diff_adj` | (Playoff BPM − RS BPM) × min(playoff_games/10, 1). Games coefficient dampens small-sample noise. |
| `playoff_rapm_diff_adj` | Same logic for RAPM diff. |

## Models

### Phase 1: Ridge Regression

- **StandardScaler** → **Ridge(alpha=1.0)**
- **Cross-validation**: 5-fold GroupKFold (same player stays in same fold to prevent data leakage)
- **Data**: year-1 only, rookie scale filtered out
- **NaN handling**: fill with per-column median, then 0 for all-NaN columns

| Metric | Value |
|--------|-------|
| CV R² | 0.815 ± 0.036 |
| CV MAE | 0.027 |
| Train R² | 0.827 |
| Samples | 1,216 |

### Phase 2: XGBoost

- **XGBRegressor** with GridSearchCV hyperparameter tuning
- **Data**: year-1 only, rookie scale kept as a feature (not filtered)
- **Best params**: max_depth=5, min_child_weight=5, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8

| Metric | Value |
|--------|-------|
| CV R² | 0.865 ± 0.024 |
| CV MAE | 0.016 |
| Train R² | 0.977 |
| Samples | 1,739 |

### Model Progression

| Version | Model | R² | Samples | Key Change |
|---------|-------|-----|---------|-----------|
| v1 | Ridge | 0.574 | ~400 | Single-season, composite rating |
| v2 | Ridge | 0.618 | ~400 | Separate z-scored DARKO/LEBRON/RAPM |
| v3 | Ridge | 0.622 | 2,130 | Multi-season 2019-2026, rookie filter |
| v4 | Ridge | 0.641 | 2,130 | Added BPM, shooting splits, ast_pct |
| v5 | Ridge | 0.676 | 2,130 | Fixed age source, playoff diffs, made_playoffs |
| v6 | Ridge | 0.806 | 2,130 | CBA structural features (is_vet_min, is_mle_range, is_rookie_scale) |
| v6 | XGBoost | 0.866 | 3,059 | Full data, rookie scale as feature |
| **v7** | **Ridge** | **0.815** | **1,216** | **Year-1-only filter (contract year 1 = market signal)** |
| **v7** | **XGBoost** | **0.865** | **1,739** | **Year-1-only filter** |

Key insight: year-1 filter removed 44% of rows but R² held steady or improved — confirming that year 2+ contract rows were noise (locked-in escalators, not market evaluations).

## Inference & Free Agent Identification

For the 2026-27 prediction:
- **Data requirement**: only players with complete RAPM data (age + minutes present) are included. Players with only DARKO/LEBRON data are excluded (102 players in 2025-26, mostly injured players who missed the season).
- **Under contract**: players in salary data for season ≥ 2026
- **Free agents**: players with 2025-26 impact metrics (season=2026) but no 2026-27+ salary entry
- **Reference salary for diff**: under-contract → actual 26-27 salary; FAs → previous year (25-26) salary

Current output: **458 players** (154 FAs).

## Known Issues & Limitations

### 1. CBA Feature Leakage at Inference
`is_vet_min`, `is_mle_range`, `is_rookie_scale` are derived from `cap_pct` — the target variable itself. During training, they provide strong signal about contract structure (R² jumped from 0.676 → 0.806 when added). But at inference:
- For **free agents**, there is no current cap_pct → these features are unavailable (default to 0)
- For **under-contract players**, computing them from actual salary is circular
- **Consequence**: the model trains with these features but predicts without them, creating a train/inference mismatch

### 2. Rookie / Young Star Underprediction
Players like Wembanyama (RAPM 8.16, BPM 10.7 — MVP-level stats) are predicted at only ~$22M. Root causes:
- The training data associates young players with low `cap_pct` because most young players ARE on rookie-scale or early contracts
- Without contract-type features at inference, the model relies on age + minutes, which penalize young players
- Low minutes (1866 for Wembanyama vs 2500+ for established stars) further suppresses the prediction, even when per-minute production is elite

### 3. Minutes as Both Signal and Noise
Minutes played is a strong predictor (more playing time → higher salary) but conflates:
- **Role / status** (stars play 35+ mpg — this is signal)
- **Health / missed games** (injury-shortened seasons lower total minutes — this is noise for market value)
- A player who dominates in 50 games looks worse than a decent player in 82 games

### 4. Supermax / Legacy Contract Distortion
Players on supermax deals (Curry $62M, Tatum $58M, Embiid $58M) have negative diffs of -$15M to -$21M. These contracts reflect:
- Designated veteran extensions (8% annual raises vs 5%)
- Bird rights retention premium
- Max eligible % increases that compound over contract length

The model can't reach these values because few training examples exist at cap_pct > 0.30.

### 5. Incomplete Data Coverage
- **102 players** missing from 2026 predictions due to no RAPM data (injured/DNP players)
- **Team wins** matched for only 2,530/3,059 training rows
- **Playoff features** have ~35% coverage; non-playoff players get median-filled values
- **Historical salary data** (pre-2024) is incomplete — BBRef only shows currently active contracts, so expired contracts are missing from team payroll calculations

## Planned Improvements

- **Separate market-value target**: filter training data to only include contracts signed in that season (not continuing contracts), so the model learns "what did teams actually offer" rather than "what is the player currently being paid"
- **Per-minute or per-36 normalization**: reduce sensitivity to total minutes / games played
- **Apron status**: whether the signing team was above first/second apron (constrains MLE, sign-and-trade options). Requires historical team payroll data not currently available.
- **Two-stage model**: (1) classify expected contract type (vet min / MLE / mid-tier / max), then (2) predict cap_pct within each tier
- **SHAP analysis**: TreeExplainer for per-prediction feature attribution
