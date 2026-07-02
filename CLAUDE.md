# CLAUDE.md — NBA Free Agent Valuation Model

## Project Overview

Build an open-source, reproducible NBA free agent valuation model inspired by Hollinger's BORD$. The model predicts a player's market value as **% of salary cap**, using publicly available data and open advanced metrics as the base rating.

The key differentiator: unlike BORD$, this model is fully transparent, reproducible, and incorporates CBA structural constraints as explicit features.

## Architecture

```
nba-valuation/
├── CLAUDE.md
├── requirements.txt
├── config.py                # seasons, cap values, CBA params, feature toggles
├── src/
│   ├── scraping/
│   │   ├── contracts.py     # Spotrac / HoopsHype: historical contract data
│   │   ├── stats.py         # Basketball Reference: per-game, advanced, per-100
│   │   ├── advanced.py      # EPM / LEBRON / DARKO (if API available)
│   │   └── availability.py  # games played history, injury logs
│   ├── features/
│   │   ├── base_rating.py   # ingest EPM/LEBRON, normalize across seasons
│   │   ├── age_curve.py     # age-based growth/decline adjustment
│   │   ├── availability.py  # weighted GP% over past 3 seasons
│   │   ├── cba_constraints.py  # Bird rights, apron status, max eligible %, signing type
│   │   └── build_dataset.py # merge all features into training-ready DataFrame
│   ├── model/
│   │   ├── train.py         # XGBoost / Ridge / Lasso with CV
│   │   ├── evaluate.py      # residual analysis, prediction vs actual plots
│   │   └── predict.py       # inference on upcoming free agents
│   └── analysis/
│       ├── feature_importance.py
│       └── residual_analysis.py  # find systematic over/underpay patterns
├── data/
│   ├── raw/                 # scraped CSVs (gitignored)
│   ├── processed/           # cleaned, merged datasets
│   └── reference/           # cap history, CBA rule tables (manually curated)
├── notebooks/               # exploratory analysis, visualization
│   └── eda.ipynb
└── outputs/
    ├── models/              # saved model artifacts
    └── predictions/         # free agent valuation outputs
```

## Data Sources (all public)

| Source | Data | Format |
|--------|------|--------|
| Basketball Reference | box score, advanced stats, per-100, GP history | HTML scrape |
| Spotrac / HoopsHype | contract details ($/year, years, signing type) | HTML scrape |
| Dunks & Threes (dunksandthrees.com) | EPM data | HTML/CSV |
| DARKO (apanacea.com) | DARKO projections | possibly API |
| LEBRON (BBall Index) | LEBRON metric | check availability |
| NBA.com/stats | tracking data (optional, stretch goal) | JSON API |
| Manual reference | cap history by year, CBA rule changes | hand-curated CSV |

## Target Variable

`cap_pct` = contract annual salary / salary cap in that signing year

Use **all active contracts** each season (not just new signings), so every player-season is one row. This maximizes sample size (~400+ per season × 5-6 seasons = 2000-2500 rows).

## Feature Set (Phase 1 — Core)

| Feature | Source | Notes |
|---------|--------|-------|
| `base_rating` | EPM or LEBRON | primary performance signal |
| `age` | BBREF | at time of contract/season |
| `age_squared` | derived | capture nonlinear decline |
| `availability_3yr` | BBREF GP | weighted avg GP% (weights: 0.5 / 0.3 / 0.2) |
| `position` | BBREF | one-hot or ordinal |
| `usage_rate` | BBREF advanced | offensive role proxy |
| `experience_years` | BBREF | years in league |

## Feature Set (Phase 2 — CBA Constraints)

| Feature | Source | Notes |
|---------|--------|-------|
| `signing_type` | Spotrac | Bird / Early Bird / Cap Space / MLE / Min / etc. |
| `team_apron_status` | derived | is signing team above 1st/2nd apron? |
| `max_eligible_pct` | CBA rules + experience | 25% / 30% / 35% of cap |
| `is_rookie_scale` | Spotrac | boolean |
| `cba_era` | manual | pre-2023 vs post-2023 CBA |

## Feature Set (Phase 3 — Stretch Goals)

| Feature | Source | Notes |
|---------|--------|-------|
| `archetype_cluster` | K-Means on play style data | only add if residual analysis shows fit matters |
| `team_positional_need` | derived from roster minutes | vacancy proxy |
| `prior_team_retained` | Spotrac | boolean: did player re-sign? (Bird rights premium) |

## Modeling Strategy

### Phase 1: Baseline
- Ridge regression with core features only
- 5-fold CV, evaluate R², MAE, MAPE
- Sanity check: do the predictions pass the smell test for known players?

### Phase 2: Gradient Boosted Trees
- XGBoost with all Phase 1 + Phase 2 features
- Hyperparameter tuning via Optuna or GridSearchCV
- Feature importance analysis (SHAP values preferred)
- Compare to Ridge baseline — is the complexity justified?

### Phase 3: Residual Analysis & Iteration
- Plot residuals by archetype, team, signing type
- If systematic patterns emerge (e.g., rim-running bigs consistently overpaid), consider adding targeted features
- **Do NOT add clustering / fit features unless residual analysis justifies it**

## Coding Conventions

- Python 3.10+
- Use `pandas` for data manipulation, `scikit-learn` for modeling, `xgboost` for GBT
- `matplotlib` / `seaborn` for viz
- Type hints encouraged but not mandatory
- Docstrings for all functions in `src/`
- Scraping: use `requests` + `BeautifulSoup`, respect rate limits (3s delay between requests), cache raw HTML to avoid re-scraping
- All dollar amounts stored as `cap_pct` (float 0-1), never raw dollars

## Immediate Next Steps

1. Set up the repo: `requirements.txt`, `config.py`, directory structure
2. Build `contracts.py` scraper — get historical contract data from Spotrac/HoopsHype for 2019-2025 seasons
3. Build `stats.py` scraper — get advanced stats from Basketball Reference for the same window
4. Build `availability.py` — compile games played history
5. Implement `build_dataset.py` to merge everything into one clean DataFrame
6. Train Ridge baseline, evaluate, iterate

## Important Notes

- **Do not overfit**: with ~2000 rows, keep model complexity in check. Ridge/Lasso before XGBoost, XGBoost before NN. Justify each step up in complexity with CV improvement.
- **Cap % normalization**: always convert raw dollars to cap %, never mix eras without normalization.
- **CBA regime awareness**: the 2023 CBA changed contract structure significantly (second apron, restrict trade rules). At minimum include a binary `cba_era` flag; ideally encode the actual constraints.
- **Avoid double counting**: if availability is already reflected in minutes projection, don't apply a separate availability discount on top.
- **Scraping courtesy**: cache everything, rate-limit requests, don't hammer servers.
