# Version History — NBA Free Agent Valuation Model

## Phase 1: Ridge exploration (single-season → multi-season)

| Ver | Model | CV R² | N | Feat | Change |
|-----|-------|-------|---|------|--------|
| 1.0 | Ridge | 0.429 | 297 | 4 | First model. Single-season (2025 only), composite `base_rating`, age, availability, usage |
| 1.1 | Ridge | 0.574 | 297 | 8 | Dropped composite rating → separate z-scored DARKO/LEBRON/RAPM. Added minutes, height |
| 1.2 | Ridge | 0.618 | 297 | 9 | Added experience_years, cba_era, max_eligible_pct. Dropped WAR (double-counts rating × minutes) |
| 2.0 | Ridge | 0.641 | 1222 | 11 | Multi-season training (2019-2025, ~4x data). Historical salary scrape from BBRef |
| 2.1 | Ridge | 0.635 | 1947 | 12 | Extended to 2019-2026. Added BPM, shooting splits. Diminishing returns from box-score stats |
| 2.2 | Ridge | 0.622 | 2874 | 12 | Full dataset with all team contracts. More data diluted signal slightly |

## Phase 2: Feature engineering + XGBoost (leaked metrics era)

| Ver | Model | CV R² | N | Feat | Change |
|-----|-------|-------|---|------|--------|
| 3.0 | Ridge | 0.642 | 2890 | 18 | Added offense/defense splits, playoff features, years_from_peak. Kitchen sink approach |
| 3.1 | Ridge | 0.676 | 2130 | 19 | Age-heuristic rookie scale filter (removed ~760 rookie rows). Cleaner training signal |
| 3.2 | Ridge | 0.676 | 2130 | 20 | Tested ast_pct, ast_tov_ratio — marginal. Cleaned redundant CBA columns |
| 3.3 | Ridge | 0.676 | 2130 | 22 | Added win_pct, made_playoffs — near-zero impact |
| 3.4 | Ridge | 0.724 | 2130 | 23 | Added `is_vet_min`, `is_mle_range` — **leaked features** (derived from target). +0.05 was illusory |
| 3.4x | XGBoost | 0.759 | 2130 | 23 | First XGBoost attempt. Big gain over Ridge on same features |
| 3.5 | Ridge | 0.806 | 2130 | 24 | Added `is_rookie_scale` as feature — also leaked. Ridge ceiling |
| 3.5x | XGBoost | 0.866 | 3059 | 25 | XGBoost on full dataset with all leaked features. Highest R² but meaningless |

## Phase 3: Data cleanup + year-1 filter

| Ver | Model | CV R² | N | Feat | Change |
|-----|-------|-------|---|------|--------|
| 4.0 | Ridge | 0.645 | 1739 | 11 | Year-1 filter (drop CBA escalator rows). Removed all leaked features. Clean 11-feature set |
| 4.1 | Ridge | 0.671 | 1739 | 11 | Added `mpg` (minutes/games) instead of raw minutes. Better signal for per-game workload |
| 4.1x | XGBoost | 0.707 | 1216 | 11 | XGBoost + age-heuristic rookie filter. Overfit on small N |
| 4.2x | XGBoost | 0.732 | 1487 | 11 | Replaced age-heuristic rookie filter with draft_data.csv-based filter. More precise, +271 rows |

## Phase 4: Ablation + hyperparameter tuning

| Ver | Model | CV R² | N | Feat | Change |
|-----|-------|-------|---|------|--------|
| 5.0 | Ridge | 0.659 | 1487 | 13 | Systematic ablation: removed 16 rejected candidates (BPM, shooting, playoff, agent features). 13 clean features |
| 5.0x | XGBoost | 0.749 | 1487 | 13 | XGBoost on ablated feature set. Default hyperparams |
| 5.1x | XGBoost | 0.749 | 1487 | 13 | Phase 1 grid search: depth × n_estimators × lr (64 combos × 10 seeds). Best: depth=4, n_est=500, lr=0.01 |
| 5.2x | XGBoost | 0.751 | 1487 | 13 | Phase 2 grid search: mcw × subsample × colsample (27 combos × 10 seeds). +regularization: mcw=10, sub=0.7, col=0.7 |
| 5.3x | XGBoost | 0.751 | 1487 | 13 | Fixed award forward-fill bug (cumulative score wasn't persisting to non-award seasons) |

## Phase 5: Feature additions (clean evaluation)

All R² values in this phase use 10-seed means.

| Ver | Model | 10-seed CV R² | 10-seed Holdout R² | N | Feat | Change |
|-----|-------|---------------|-------------------|---|------|--------|
| 6.0x | XGBoost | 0.7583 ± 0.0009 | 0.8289 ± 0.0029 | 1487 | 14 | +`prev_cap_pct` (previous contract anchoring). +0.008 CV vs 13-feature baseline |
| 6.1x | XGBoost | 0.7575 ± 0.0008 | 0.8206 ± 0.0037 | 1487 | 14 | Comprehensive awards (Wikipedia). Exp decay 0.85/yr. +All-Star/Starter. −All-Star MVP. Slightly worse than 6.0x |
| **6.2x** | **XGBoost** | **0.7598 ± 0.0009** | **0.8341 ± 0.0038** | **1487** | **14** | **Award ablation: awards_full 2017-26, −All-Star, step decay [1, 0.85, 0.65, 0.4, 0.1]. +0.0015 CV vs v6.0x** |

v6.2x award ablation results (all 10-seed, awards_full.csv + step decay):
- K (adopted): 2017-26, −All-Star → CV 0.7598, HO 0.8341 ← best
- J: 2017-26, +All-Star → CV 0.7584, HO 0.8312
- G: 2014-26, +All-Star → CV 0.7575, HO 0.8281
- I: old award.csv, step decay → CV 0.7587, HO 0.8316

## Key milestones

```
Phase 1    Phase 2 (leaked)     Phase 3         Phase 4        Phase 5
Ridge      Ridge/XGB            Cleanup         Tuning         Features
                                                               
0.43 ─→ 0.62 ─→ 0.68 ─→ ···    0.65 ─→ 0.73    0.75 ─→ 0.75   0.76
 v1.0     v2.2    v3.1  ···      v4.0    v4.2x    v5.0x   v5.3x   v6.2x
                        ↗ 0.87                                    (current)
                  Leaked features
                  (removed in v4.0)
```

## Notes

**R² comparability**: Phase 1–2 (v1.0–v3.5x) and Phase 3+ (v4.0+) use different training data filters and are not directly comparable. The apparent drop from v3.5x (0.866) to v4.0 (0.645) reflects removing leaked features and applying stricter filters — not a regression.

**Phase 2 leaked features**: `is_vet_min`, `is_mle_range`, `is_rookie_scale` are derived from the target variable (salary determines contract type). They gave large R² gains in training but are unknowable at prediction time. All removed in v4.0.

**v6.2x award design**: Step decay [1, 0.85, 0.65, 0.4, 0.1] outperforms exp decay (0.85/yr) and no-decay across all data sources. Filtering to 2017+ removes noisy older data. All-Star selections add noise (−0.0014 CV); team awards (All-NBA, All-Defensive, All-Rookie) and individual awards are the useful signal.
