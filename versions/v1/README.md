# v1 — Baseline Model Snapshot

Frozen: 2026-06-27

## Model
- **XGBoost** CV R² = 0.7317 (5-fold GroupKFold)
- **Ridge** CV R² = 0.7071
- Training rows: 1,487 (after year-1 + draft-based rookie filter)
- 657 unique players, 8 seasons (2019–2026)

## Features (11)
1. darko_dpm_z
2. lebron_z
3. rapm_z
4. age
5. age_squared
6. mpg
7. availability_3yr
8. usage_pct
9. height_inches
10. cba_era
11. ast_pct

## Filters
- **Year-1 filter**: keep only year_in_contract == 1 (3,059 → 1,739)
- **Rookie scale filter**: draft-based, removes 1st-round picks at seasons draft_year+1 to +3 using draft_data.csv (1,739 → 1,487)

## Holdout Results
- 2026 season holdout: R² = 0.7148, MAE = $4.9M (55 test samples)
- 200-player random holdout: R² = 0.7835, MAE = $4.4M (448 test rows)

## Key Files
- `src/model/train.py` — training pipeline with FEATURE_COLS
- `src/model/predict.py` — inference for target season
- `src/features/build_dataset.py` — data merging and feature engineering
- `config.py` — season/cap constants
- `outputs/predictions/predictions_2026_v9.csv` — 458 players
- `outputs/predictions/predictions_2026_shap.csv` — per-feature SHAP impacts
- `outputs/predictions/holdout_200_kmeans.csv` — K-means clustered holdout
