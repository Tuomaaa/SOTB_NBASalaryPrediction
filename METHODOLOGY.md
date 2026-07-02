# Methodology — NBA Free Agent Valuation Model

## Objective

Predict a player's market value as **cap_pct** (annual salary / salary cap), using prior-season performance data. Inspired by Hollinger's BORD$, but fully open-source and reproducible.

## Data Sources

| Source | Data | Rows |
|--------|------|------|
| Basketball Reference | Per-season salary, age, team | 4,397 player-seasons (1,037 players, 2019–2030) |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage, box-score rates | 3,880 player-seasons (903 players) |
| Basketball Reference | Height in inches | 5,416 players |
| Spotrac | Free agent signings, signing type (Bird/MLE/cap space/etc.) | 981 FA signings (2019–2025) |
| Manual reference (raw_external/) | Awards, draft position, team franchise value | Hand-curated CSVs |
| Manual reference | Salary cap by year, CBA parameters | config.py |

## Season Convention & Intentional Lag

Salary data uses the **start year** (season=2025 → 2025-26 salary) while nbarapm.com uses the **end year** (season=2025 → 2024-25 stats). When merged on `season`, this creates a **1-year lag**: prior-season stats predict current-season salary. This is intentional — teams sign contracts based on prior performance.

## Target Variable

```
cap_pct = annual_salary / salary_cap_for_that_season
```

Range: 0.0 to ~0.35 (max contract). All dollar amounts normalized to cap percentage for cross-era comparison.

## Training Data Construction

1. **Inner join** salaries × impact metrics on `(player_name_norm, season)` → 3,059 matched rows
2. **Merge** height, awards (cumulative with forward-fill), draft position, team value
3. **Contract structure detection**: salary progression analysis detects contract boundaries. Each row tagged with `year_in_contract` and `contract_years`.
4. **Year-1 filter**: only keep `year_in_contract == 1` rows. Years 2+ are CBA-mandated escalators (5%/8% raises), not market evaluations. Reduces to **1,739 rows**.
5. **Rookie scale filter**: remove 1st-round picks in years 2-4 of rookie deal (slotted by draft position, not market). Uses draft_data.csv for precise identification. Reduces to **1,487 rows**.
6. **GroupKFold CV**: 5-fold, same player stays in same fold to prevent within-player leakage.

## Feature Set (v3 — 14 features)

### Performance Metrics (z-scored within season)
| Feature | Description |
|---------|-------------|
| `darko_dpm_z` | DARKO Daily Plus-Minus, z-scored by season |
| `lebron_z` | LEBRON metric, z-scored by season |
| `rapm_z` | Regularized Adjusted Plus-Minus, z-scored by season |

### Age & Workload
| Feature | Description |
|---------|-------------|
| `age` | Player age at time of season |
| `age_squared` | Captures nonlinear decline after peak |
| `mpg` | Minutes per game (minutes / games) |
| `usage_pct` | Usage rate (offensive load) |
| `availability_3yr` | Weighted GP% over past 3 seasons (0.5/0.3/0.2) |

### Physical & Draft
| Feature | Description |
|---------|-------------|
| `height_inches` | Player height from BBRef |
| `draft_pick` | Overall draft pick number (undrafted=75) |

### Awards & Reputation
| Feature | Description |
|---------|-------------|
| `award_score_cum` | Cumulative award score (MVP=10, FMVP=8, DPOY=5, All-NBA 1st=4, etc.). Forward-filled across all seasons so cumulative value persists even in non-award years. Award Year Y maps to season Y+1 (lagged — captures reputation premium). |
| `ast_pct` | Assist percentage (playmaking proxy) |

### CBA & Contract History
| Feature | Description |
|---------|-------------|
| `cba_era` | Binary: 0 = pre-2023 CBA, 1 = post-2023 CBA |
| `prev_cap_pct` | Year-1 cap_pct of the player's previous contract. Captures anchoring effect — prior contract value predicts next contract. Filled with median rookie-scale cap_pct for first contracts. |

### Features Tested and Rejected

| Feature | ΔCV R² | Reason Rejected |
|---------|--------|-----------------|
| `bpm`, `ts_plus`, `fg3_plus`, `threepar_plus` | < +0.001 | Redundant with DARKO/LEBRON/RAPM |
| `ast_tov_ratio` | +0.0001 | Redundant with ast_pct |
| `experience_years` | -0.0004 | Collinear with age |
| `win_pct`, `made_playoffs` | +0.0004 | Team context already in player metrics |
| `playoff_bpm_diff_adj`, `playoff_rapm_diff_adj` | +0.0003 | Sparse (35% coverage), noisy |
| `max_eligible_pct` | +0.0008 | Collinear with age/experience |
| `is_vet_min`, `is_mle_range` | N/A | Derived from target variable (leakage) |
| `is_rookie_scale` | N/A | Handled by rookie scale filter instead |
| `team_value_B` | +0.0001 | Franchise value doesn't predict individual salary |
| `injury_reports` | +0.0002 | Redundant with availability_3yr |
| `is_lottery`, `is_top5` | < +0.001 | Redundant with draft_pick |
| `all_nba_cum` | +0.0001 | Redundant with award_score_cum |
| Agent portfolio features | -0.002 to -0.005 | Leakage when computed naively; too sparse for native categorical (1,487 rows / 147 agents) |
| `team_cap_space_pct` | +0.0008 | Team's cap situation doesn't predict individual contract value |
| `team_over_cap` | +0.0003 | Same as above, binary version |
| `is_contract_year` | +0.0009 | Contract year effect not significant in data |
| `stayed_with_team` | +0.0006 | Bird rights proxy from Spotrac; near-zero correlation with cap_pct |

## Model — XGBoost v3

```python
XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    tree_method="hist",
)
```

Hyperparameters found via two-phase grid search (Phase 1: depth×n_est×lr, 64 combos; Phase 2: mcw×sub×col, 27 combos; 10 seeds each). Direction: more weak learners + lower learning rate + stronger regularization = better generalization.

### Results

| Metric | v2 (13 features) | v3 (14 features) |
|--------|-------------------|-------------------|
| CV R² | 0.7504 | **0.7588** |
| 2024 Holdout R² | 0.8337 | **0.8385** |
| 2025 Holdout R² | 0.8255 | **0.8294** |
| 2026 Holdout R² | 0.8085 | **0.8266** |
| 2026 Holdout MAE | $3.9M | **$3.8M** |
| Train samples | 1,487 | 1,487 |

### Model Progression

| Version | Model | CV R² | Key Change |
|---------|-------|-------|------------|
| v1 | Ridge | 0.574 | Single-season, composite rating |
| v2 | Ridge | 0.618 | Separate z-scored DARKO/LEBRON/RAPM |
| v3 | Ridge | 0.622 | Multi-season 2019-2026 |
| v4 | Ridge | 0.641 | Added BPM, shooting splits |
| v5 | Ridge | 0.676 | Fixed age source, playoff features |
| v6 | Ridge | 0.806 | CBA structural features (is_vet_min, is_mle_range) |
| v6 | XGBoost | 0.866 | Full data, rookie scale as feature |
| v7 | Ridge | 0.815 | Year-1 filter (contract year 1 = market signal) |
| v7 | XGBoost | 0.865 | Year-1 filter |
| v8 (Phase 2) | XGBoost | 0.7505 | 13 features after ablation, tuned hyperparams, award forward-fill fix |
| **v3 (current)** | **XGBoost** | **0.7588** | **+prev_cap_pct, 14 features** |

Note: v8+ CV R² numbers appear lower than v6/v7 because of the rookie scale filter change (from age-based heuristic to draft_data.csv-based filter) and stricter year-1 filtering. The model is more accurate on market-priced contracts.

## Holdout vs Valuation

Two distinct evaluation modes:

- **Holdout**: year-1 + rookie filter applied to BOTH train and test. Used for model evaluation (R², MAE). Apples-to-apples comparison with training distribution.
- **Valuation**: year-1 + rookie filter on train only, score ALL rows. Used to find overpaid/underpaid contracts. R² not meaningful here — the goal is ranking and diffs.

## Diagnostic Findings (v3, 2026 holdout)

### Residual Patterns
- **High earners (25%+ cap)**: systematically underpredicted by ~$11.8M. Regression to mean + supermax contracts the model can't reach.
- **Young players (21-24)**: well-predicted (MAE $3.0M, near-zero bias).
- **Older players (25+)**: underpredicted by $4-5M — veteran max extensions exceed what stats alone justify.
- **Tall players (80"+)**: underpredicted by $3.3M — max-contract bigs (JJJ, Bam, Chet).

### Biggest Misses
Top misses are almost all underpredictions of max/near-max contracts: Jaren Jackson Jr. (-$26M), Jalen Williams (-$21M), Jabari Smith Jr. (-$14M). These are rookie max extensions where the contract reflects projected ceiling, not current production.

## Inference Pipeline

`src/model/predict.py`:
1. Load training data, apply year-1 + rookie filters, train XGBoost
2. Load impact_metrics for target season
3. Engineer features (base rating, age, availability, CBA, height, awards, draft_pick, prev_cap_pct)
4. Predict cap_pct → convert to salary
5. Identify free agents (have stats but no salary for target season)
6. Output: predictions CSV + free agents CSV

## Known Issues & Limitations

1. **Supermax ceiling**: model rarely predicts cap_pct > 0.28. Training data has few examples at 30%+, and tree models don't extrapolate.
2. **Rookie max extensions**: players like JJJ, Jalen Williams get max extensions based on projected ceiling — the model sees current stats only.
3. **prev_cap_pct fill**: first-contract players get median rookie-scale cap_pct as fill value. Could be improved with draft pick → expected rookie scale mapping.
4. **Signing type**: Spotrac player pages contain explicit signing mechanism (Bird Rights, MLE, cap space, etc.). Scraping in progress — if coverage is sufficient, will test as Phase 3C feature.
5. **No tracking data**: NBA.com tracking data (drives, catch-and-shoot, rim protection) could improve archetype-specific predictions.

## Reproducibility

```bash
# Rebuild features
python scripts/build_external_features.py

# Train and evaluate
python src/model/train.py

# Generate 2026 predictions
python src/model/predict.py

# Run diagnostics
python scripts/diagnostics.py
```
