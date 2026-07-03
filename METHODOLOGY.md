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

## Model — Two-Stage Pipeline (v7.0x)

### Stage 1: Grabit (Nonlinear Tobit via XGBoost)

```python
XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    tree_method="hist",
    objective=make_tobit_obj(censored_mask, sigma=0.02),
)
```

Standard XGBoost treats all training rows equally. But max-contract players' observed salary is a **ceiling**, not their true market value — a player worth 40% of the cap still gets paid 35% because that's the CBA maximum. Standard squared-error loss trains the model to predict 35%, permanently underpredicting elite players.

**Grabit** (Sigrist & Hirnschall, 2019) replaces XGBoost's loss function with a censored-normal likelihood:
- **Uncensored rows** (non-max contracts): standard squared error. `grad = pred - actual`, `hess = 1`.
- **Censored rows** (max contracts): the loss says "the true value is *at least* this much." `grad = -σ × m(z)`, `hess = m(z) × (z + m(z))`, where `m(z) = φ(z)/Φ(z)` is the inverse Mills ratio. This pushes predictions *above* the observed ceiling.

The model outputs a **latent value** — what the player would earn in a CBA-free market.

**Gated censoring**: not all max contracts are true max-value players. Albatross contracts (John Wall 2020, Gordon Hayward 2023) are players paid the max despite declining performance. Censoring these would corrupt the signal. We only censor max-contract rows where the baseline XGBoost prediction ≥ 55% of max_eligible — filtering out ~5 albatross rows per run.

### Stage 2: CBA Cap

```
final_prediction = min(latent_value, max_eligible_pct)
```

CBA rules cap maximum salary by experience:
- **0–6 years**: 25% of cap (or 30% with Rose Rule)
- **7–9 years**: 30% of cap (or 35% with Supermax)
- **10+ years**: 35% of cap

**Rose Rule**: players with ≤6 years experience who made an All-NBA team, won MVP, or won DPOY in the current or recent seasons qualify for 30% max. Detection uses non-lagged elite award lookup directly from awards data (the award that triggers the extension happens the same season the salary starts).

**Supermax**: players with 7-9 years experience who earned multiple elite awards in a 3-year window qualify for 35% max.

### Hyperparameters

Base XGBoost hyperparameters found via two-phase grid search (Phase 1: depth×n_est×lr, 64 combos; Phase 2: mcw×sub×col, 27 combos; 10 seeds each).

Tobit-specific:
- **σ = 0.02**: controls the balance between censored and uncensored gradients. CV-optimal across grid [0.02, 0.04, 0.06, 0.10]. Larger σ improves holdout but risks overfitting.
- **Censoring gate threshold = 0.55**: baseline prediction must reach 55% of max_eligible to be censored.

### Results

| Metric | v6.2x (baseline XGB) | v7.0x (Grabit + CBA cap) |
|--------|---------------------|-------------------------|
| 10-seed CV R² | 0.7598 ± 0.0009 | **0.7612 ± 0.0012** |
| 10-seed CV R² (2024-26) | 0.8491 ± 0.0016 | **0.8523 ± 0.0016** |
| 10-seed HO R² | 0.8341 ± 0.0038 | **0.8382 ± 0.0040** |
| Total MAE | $3.2M | **$3.1M** |
| Max contract MAE | $6.7M | **$6.5M** |
| Train samples | 1,487 | 1,487 |
| Features | 14 | 14 |

**CV R² (2024-26)**: GroupKFold CV computed on the subset of OOF predictions where season ≥ 2024 (n=385). This metric better reflects future prediction ability — later seasons have higher target variance and more complete data coverage. Reported alongside full-year CV.

Note: v6.2x and earlier CV R² numbers appear lower than Phase 2 (v3.x) because of the rookie scale filter change (from age-based heuristic to draft_data.csv-based filter) and stricter year-1 filtering. The model is more accurate on market-priced contracts.

## Holdout vs Valuation

Two distinct evaluation modes:

- **Holdout**: year-1 + rookie filter applied to BOTH train and test. Used for model evaluation (R², MAE). Apples-to-apples comparison with training distribution.
- **Valuation**: year-1 + rookie filter on train only, score ALL rows. Used to find overpaid/underpaid contracts. R² not meaningful here — the goal is ranking and diffs.

## Diagnostic Findings (v7.0x, 2026 holdout)

### Residual Patterns by Signing Type
- **Bird Rights** (n=58): bias -$2.7M, MAE $4.9M. Underpredicted — Bird rights premium the model can't observe.
- **Sign & Trade** (n=17): bias -$2.3M, MAE $6.1M. Same retention/premium mechanism.
- **Cap Space** (n=51): bias -$0.8M, MAE $4.2M. Slight underprediction.
- **MLE** (n=61): bias +$1.4M, MAE $2.5M. Overpredicted — model thinks MLE players deserve more than MLE cap allows.
- **Minimum** (n=419): bias +$1.5M, MAE $2.4M. Overpredicted — vets taking minimums are better than their salary; model sees stats, not mechanism.
- **Early Bird** (n=27): bias -$0.4M, MAE $2.7M. Roughly calibrated.

### Biggest Misses
Top misses are max-contract underpredictions: Jaren Jackson Jr. (-$26M, injured season), Jalen Williams (-$21M, max extension on projected ceiling), Jabari Smith Jr. (-$13M), De'Aaron Fox (-$12M). These are cases where the contract reflects projected upside or franchise commitment, not current-season production.

### Grabit Impact on Max Contracts
The Grabit censored loss improved predictions for max-contract players across all three CBA tiers (25%, 30%, 35%). Out of 73 max-contract player-seasons, 58 improved, 7 worsened, 8 neutral. The 7 worsened cases are albatross contracts (gating catches most but not all) and edge cases where Tobit slightly overpushes.

## Inference Pipeline

`src/model/predict.py`:
1. Load training data, apply year-1 + rookie filters
2. Compute max_eligible_pct with Rose Rule / Supermax detection
3. Train baseline XGBoost → compute gated censoring mask
4. Train Grabit XGBoost (censored loss) on all training data
5. Predict latent value → apply CBA cap: `min(latent, max_eligible_pct)`
6. Convert cap_pct to salary dollars
7. Output: predictions CSV + free agents CSV

## Known Issues & Limitations

1. **Rookie max extensions**: players like JJJ, Jalen Williams get max extensions based on projected ceiling — the model sees current-season stats only, not future potential.
2. **Albatross contracts**: John Wall, Gordon Hayward — paid max despite poor performance. Gated censoring filters most but not all. The model correctly says they're overpaid.
3. **prev_cap_pct fill**: first-contract players get median rookie-scale cap_pct as fill value. Could be improved with draft pick → expected rookie scale mapping.
4. **Signing type as feature**: Spotrac signing mechanism (Bird Rights, MLE, cap space) is a diagnostic label, not a model feature — it's partially determined by the contract itself.
5. **No tracking data**: NBA.com tracking data (drives, catch-and-shoot, rim protection) could improve archetype-specific predictions.
6. **σ tension**: CV prefers σ=0.02 (conservative censored gradients), holdout prefers larger σ (more aggressive). Sticking with CV-optimal to avoid overfitting to a single holdout season.

## Reproducibility

```bash
# Rebuild features
python scripts/build_external_features.py

# Train and evaluate (baseline XGBoost)
python src/model/train.py

# Generate 2026 predictions (Grabit + CBA cap pipeline)
python src/model/predict.py

# Run diagnostics
python scripts/diagnostics.py
```

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*. — Censored-normal likelihood as XGBoost custom objective.
