# Methodology — NBA Free Agent Valuation Model

## Objective

Predict a player's market value as **cap_pct** (annual salary / salary cap), using prior-season performance data. Inspired by Hollinger's BORD$, but fully open-source and reproducible.

## Data Sources

| Source | Data | Rows |
|--------|------|------|
| Basketball Reference | Per-season salary, age, team | 4,723 player-seasons (1,086 players, 2019–2031) |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage, box-score rates | 3,880 player-seasons (903 players) |
| Basketball Reference | Height in inches | 5,416 players |
| Spotrac | Signing mechanism, contract years, total value, AAV | 2,403 contract-seasons; 981 FA signings (2019–2025) |
| Manual reference (raw_external/) | Awards, draft position, team franchise value | Hand-curated CSVs |
| Manual reference | Salary cap by year, CBA parameters | config.py |

The salary table was refreshed on 2026-07-22, after 2026-27 free agency. The
previous scrape predated it, so the 2026 season held only players already under
contract — see "Cap values are load-bearing" and "Sample composition" below.

## Season Convention & Intentional Lag

Salary data uses the **start year** (season=2025 → 2025-26 salary) while nbarapm.com uses the **end year** (season=2025 → 2024-25 stats). When merged on `season`, this creates a **1-year lag**: prior-season stats predict current-season salary. This is intentional — teams sign contracts based on prior performance.

## Target Variable

```
cap_pct = annual_salary / salary_cap_for_that_season
```

Range: 0.0 to ~0.35 (max contract). All dollar amounts normalized to cap percentage for cross-era comparison.

### Cap values are load-bearing

Because the cap is the denominator of the target, a wrong cap silently rescales
an entire season's target. Two seasons carried stale pre-media-deal projections
until 2026-07-22: 2025 was recorded as $141.208M against an actual $154.647M,
and 2026 as $153M against $166M, inflating both seasons' targets by ~9%.

The error was found from the data rather than an external source, by a method
that validates itself: dividing max contracts by 25/30/35% recovers a consensus
cap, and that consensus reproduces the configured value exactly for 2019 and
2021–2024. Two independent checks agreed — the veteran minimum rose 10.0% into
2025 (the CBA's maximum permitted increase) while the configured cap rose 0.4%,
and Luka Dončić's 2026 salary is exactly 30% of $166M. Correcting the caps moved
the calibration slope from 1.014 to **1.0006**.

`scripts/check_caps.py` now asserts this reconciliation for every season.
`load_training_data()` recomputes `cap_pct` from `salary / CAP_BY_SEASON` on
load, so `config.py` is the only place a cap is stated and a stale value baked
into a CSV cannot survive.

## Training Data Construction

1. **Inner join** salaries × impact metrics on `(player_name_norm, season)` → 3,113 matched rows
2. **Merge** height, awards (cumulative with forward-fill), draft position, team value
3. **Contract structure detection**: salary progression analysis detects contract boundaries. Each row tagged with `year_in_contract` and `contract_years`.
4. **Year-1 filter**: only keep `year_in_contract == 1` rows. Years 2+ are CBA-mandated escalators (5%/8% raises), not market evaluations. Reduces to **1,808 rows**.
5. **Rookie scale filter**: remove 1st-round picks in years 2-4 of rookie deal (slotted by draft position, not market). Uses draft_data.csv for precise identification. Reduces to **1,556 rows** across 666 players, 2019–2026.
6. **Prorated filter**: drop rows below 1.2% of that season's cap — partial-season pay, not an annual contract value. Reduces to **1,297 rows**. See below.
7. **GroupKFold CV**: 5-fold, same player stays in same fold to prevent within-player leakage.

### Sample composition

The 2026 season carried 55 usable rows before the 2026-07-22 refresh and 124
after. The pre-refresh set was not merely smaller, it was **optimistically
biased**: it contained only extensions and players already under contract, all
of which are easier to price than an open-market signing. Adding the actual free
agent signings dropped the 2026 forward R² from 0.843 to 0.811 on a test set
that finally represents the task.

### Prorated salaries (removed in v7.2x)

259 rows (17%) paid less than 1.2% of the cap, a median of $0.29M with a median
28 games played and 12.5 minutes. These are 10-day contracts and mid-season
signings — the same player at the same ability lands on a different target
depending only on the date he signed, so the row carries no market information.

The floor is a share of the cap, not a dollar figure, because the veteran
minimum tracks the cap: across 2019–2026 it sits between 1.30% and 2.35%, so
1.2% clears every full-season minimum while cutting the prorated rows.

Removing them is a **calibration fix rather than an accuracy gain**. Their
presence dragged every prediction down by about half a million dollars. On a
fixed evaluation set of the 1,297 clean rows, varying only the training set:
R² 0.7534 → 0.7588, MAE $3.289M → $3.327M, bias **−$0.423M → +$0.101M**.
Re-centring the filtered predictions by their mean shift returns R² to 0.7522 —
the conditional pricing function is unchanged, only the level moved.

## Feature Set (14 features)

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
| Agent portfolio features | -0.002 to -0.005 | Leakage when computed naively; too sparse for native categorical (147 agents over the training set) |
| `team_cap_space_pct` | +0.0008 | Team's cap situation doesn't predict individual contract value |
| `team_over_cap` | +0.0003 | Same as above, binary version |
| `is_contract_year` | +0.0009 | Contract year effect not significant in data |
| `stayed_with_team` | +0.0006 | Bird rights proxy from Spotrac; near-zero correlation with cap_pct |
| `P(Minimum \| x)`, `P(Bird \| x)` | **-0.0073** | Fold-honest classifier probabilities. See "Why signing mechanism cannot be a feature" |
| Post-hoc recalibration (linear) | -0.0013 | Nested, fold-honest. The model is already calibrated (slope 1.0006) |
| Post-hoc recalibration (isotonic) | -0.0052 | As above, and more prone to overfitting the fold |

### Why signing mechanism cannot be a feature

Conditioning on the model's own prediction, players who signed Minimum deals are
overpaid-for by $1.24M / $3.94M / $9.10M in the 2-5% / 5-10% / 10-20% predicted
bands, against roughly zero for the unlabelled majority. The effect is real and
survives controlling for predicted value, so it is tempting to model.

It is not recoverable. Signing mechanism is itself a function of the same player
features, and a gradient-boosted model already extracts everything those
features say about it — feeding back a fold-honest `P(mechanism | x)` *lowers*
CV R² by 0.0073. The gap is the spread of a genuinely bimodal `y | x`: given the
same inputs, a player either lands a market deal or takes a minimum, and the
conditional mean correctly sits between the two modes and is "wrong" for both.

An oracle run supplying the *realised* mechanism as a one-hot feature — leaky,
and an upper bound on what mechanism knowledge could ever be worth — gains only
+0.0137, and part of that is the `Unknown` category encoding whether the player
was a free agent at all. Any real gain therefore has to come from information
that is **not** a function of the current features: market supply and demand at
the position that summer, how the previous contract terminated, league-wide cap
room. Modelling the mechanism itself is a dead end for point accuracy.

## Model — Two-Stage Pipeline

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

## Evaluation Protocol

`src/model/evaluate_suite.py`. Four layers, each answering a different question.
They are not interchangeable and a number from one must never be quoted as
though it came from another.

### A — selection

Pooled 5-fold GroupKFold CV, 10 seeds, plus its 2024-26 subset. This is the only
input to accept/reject decisions.

Pooled CV is a legitimate estimator here even though a fold may train on a later
season and score an earlier one. The estimand is the **market's pricing
function** — a structural quantity, not a forecast — so using all available data
is efficient rather than optimistic. Player grouping blocks the leakage that
does matter, since one player's contracts are highly correlated. Time direction
was tested directly: at matched training-set size, training on later seasons
beat training on earlier ones by +0.029 ± 0.021 (t = 1.36, n = 4 origins), which
is not distinguishable from noise.

**Deltas must be paired by fold.** Fold-to-fold sd is 0.045; seed-to-seed sd is
0.001, forty-five times smaller. The `± 0.0009` figures in older version notes are the seed-averaging
noise, not a confidence interval on R². Comparing two independently-reported
means discards almost all the available power.

### B — forecasting

Rolling-origin: train on every season < T, score season T, for T ∈ {2024, 2025,
2026}. This is what `predict.py` does, and it is the number to quote for "how
well will this price next summer's signings."

Origins before 2024 are excluded deliberately. Their training sets run a third
to a fifth of the current one, so they measure data scarcity rather than the
model, and they sit in the pre-2023 CBA regime. The evidence: forward R² per
origin is 0.845 / 0.812 / 0.811 across 2024-26 (sd 0.019) against 0.446 / 0.771 /
0.700 / 0.708 across 2020-23 (sd 0.144), tracking training-set size.

### C — integrity

Calibration and per-segment bias, to catch a change that improves the average
while damaging a segment.

**Calibration is binned by predicted value, never by the target.** Binning
residuals by actual salary produces a steep monotone bias gradient even when the
model is perfectly calibrated — this is regression to the mean, and earlier
versions of this document reported it as a finding. The same predictions, binned
both ways:

| Band | binned by actual | binned by predicted |
|------|-----------------|---------------------|
| <2% | +$1.76M | −$0.01M |
| 8-15% | −$2.60M | −$0.51M |
| 25%+ | −$5.71M | +$0.00M |

The honest reading is the right column: the model is calibrated, with an OLS
slope of actual on predicted of 1.0006. Fold-honest recalibration confirms it by
failing to help (linear −0.0013, isotonic −0.0052).

### D — guards

- **Fixed evaluation set** whenever the training filter changes. R²'s denominator moves with the row set, so R² across different datasets is not comparable. The apparent v3.5x → v4.0 collapse from 0.866 to 0.645 is this effect, not a regression.
- **Baseline ladder**, so absolute R² is not mistaken for skill: predicting the mean 0.000, `mpg` alone **0.5873**, `mpg + prev_cap_pct` 0.6003, four features 0.7366, full 14 features 0.7581. Minutes per game alone reaches 77% of the full model's R²; the remaining ten features together buy +0.021 over that four-feature model.
- **Locked confirmation split**: 15% of players by stable hash, held out of every selection decision, opened at a version bump. Currently 0.7622 (n=224) against 0.7572 on the selection pool — no sign of the metric being overfitted by 20+ feature decisions and two hyperparameter sweeps.

### Results (v7.1x, 10 seeds, n = 1,556)

| Metric | Baseline XGBoost | Grabit + CBA cap |
|--------|-----------------|------------------|
| A1 CV R² | 0.7570 | **0.7581** |
| A2 CV R² (2024-26, n=454) | 0.8234 | **0.8257** |
| B1 forward R² (2024-26) | 0.8227 | **0.8249** |
| B1 95% CI | [0.771, 0.864] | [0.770, 0.867] |
| CV MAE | $3.17M | **$3.14M** |
| Calibration slope | 1.0145 | **1.0006** |
| Spearman | 0.7677 | 0.7668 |

**Paired delta (Grabit − baseline): +0.0011 ± 0.0004, t = +2.72.** Passes the
acceptance rule; the gain is small because gated censoring touches only ~73 of
1,556 rows.

Forward R² by origin (Grabit): 2024 → 0.845 (n=166), 2025 → 0.812 (n=164),
2026 → 0.811 (n=124). The cost of the forecasting setup relative to A2 is
−0.0008.

### Acceptance rule

A change is accepted when all four hold:

1. paired A1 delta > 0 with |Δ| / SE > 2
2. A2 moves the same direction (significance not required)
3. no C2 segment's bias worsens by more than $0.3M
4. MAE does not regress beyond the agreed tolerance

## Holdout vs Valuation

Two distinct scoring modes, and CONTEXT.md gives them separate names because
they mean different things:

- **Signing Board / holdout**: Year-1 filter on train and test both. Produces the **Signing Residual** — a measure of model accuracy against a price the market actually set.
- **Value Board / valuation**: Year-1 filter on train only, score every row. Produces the **Contract Surplus** — a statement about a team's books, not about model error. R² is not meaningful here; ranking is.

## Diagnostic Findings (v7.1x)

### Bias by predicted band

| Predicted band | n | bias | MAE |
|---------------|---|------|-----|
| <2% | 464 | +$0.02M | $0.89M |
| 2-4% | 373 | +$0.39M | $2.44M |
| 4-8% | 352 | −$0.03M | $4.34M |
| 8-15% | 196 | −$0.43M | $5.52M |
| 15-25% | 118 | −$0.64M | $5.64M |
| 25%+ | 56 | +$1.59M | $5.13M |

Flat to within ±$0.65M except at the very top, where n is small.

### Signing Residual by mechanism

| Mechanism | n | bias | MAE |
|-----------|---|------|-----|
| Unknown | 911 | −$0.40M | $3.36M |
| Minimum | 419 | +$1.45M | $2.35M |
| MLE | 61 | +$1.42M | $2.59M |
| Bird Rights | 58 | −$2.78M | $4.92M |
| Cap Space | 51 | −$0.68M | $4.09M |
| Early Bird | 27 | −$0.49M | $2.45M |
| Sign & Trade | 17 | −$3.04M | $6.33M |

Bird Rights and Sign & Trade are underpriced by ~$3M — both are mechanisms that
let a team exceed the cap for its own player, and the retention premium is
invisible to the features. Minimum and MLE are overpriced by ~$1.4M. These
patterns survive controlling for predicted value, but are still not usable as
features; see "Why signing mechanism cannot be a feature."

Label coverage is only 43%; the 911 `Unknown` rows carry 56% of the squared
error, so more than half of the total error cannot currently be attributed to a
mechanism at all. Roughly 7% of `Minimum` labels are also wrong — Spotrac's
contract blocks are walked backwards to assign seasons and can misalign, which
put a $27M salary under a minimum label. Correcting those makes the Minimum bias
*larger* (+$1.91M), so the effect above is understated.

### Grabit Impact on Max Contracts
The Grabit censored loss improved predictions for max-contract players across all three CBA tiers (25%, 30%, 35%). Out of 73 max-contract player-seasons, 58 improved, 7 worsened, 8 neutral. The 7 worsened cases are Albatross Contracts (gating catches most but not all) and edge cases where Tobit slightly overpushes.

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

### Modelling

1. **Rookie max extensions**: players like JJJ, Jalen Williams get max extensions based on projected ceiling — the model sees current-season stats only, not future potential.
2. **Albatross Contracts**: John Wall, Gordon Hayward — paid max despite poor performance. Gated censoring filters most but not all. The model correctly says they're overpaid.
3. **prev_cap_pct fill**: first-contract players get median rookie-scale cap_pct as fill value. Because that median is a global statistic it moves whenever rows are added — the 2026 refresh shifted it from 0.0704 to 0.0686. Could be replaced with a draft pick → expected rookie scale mapping, which would be stable.
4. **Signing mechanism is a label, not a feature** — partly determined by the contract itself, and empirically worthless as a modelled probability. See the dedicated section above.
5. **No tracking data**: NBA.com tracking data (drives, catch-and-shoot, rim protection) could improve archetype-specific predictions.
6. **σ tension**: CV prefers σ=0.02 (conservative censored gradients), holdout prefers larger σ (more aggressive). Sticking with CV-optimal to avoid overfitting to a single holdout season.
7. **Point estimates against a bimodal target**: `y | x` is genuinely bimodal for mid-market players. A conditional mean is the R²-optimal point estimate and is nonetheless wrong for both modes. A distributional output — quantiles, or `P(signs for the minimum)` alongside a conditional market value — would answer the actual question better, and would not be measurable by R².
8. **The model is not underfitting**: depth 6, learning rate 0.03, and looser `min_child_weight` all score worse than the current settings (0.7648 / 0.7613 / 0.7527 against 0.7661). Added structure cannot be justified as an inductive-bias fix.

### Pipeline and reproducibility

9. **The contract-structure script was never committed.** Only `contract_structure_v2.csv` survives. Reconstructing the detection from CBA escalator ratios reaches at best 88% agreement on the year-1 flag and 74% on contract length across a 30-point parameter sweep — far too low to regenerate history without invalidating every published version number. The table is therefore extended incrementally: rows whose salary is unchanged keep their assignment byte-for-byte, and only new or changed rows are assigned. Any future change needing a full recompute will hit this wall.
10. **Regenerating the training data spans three files with no chaining entry point**: `src/features/build_dataset.py` → `scripts/build_external_features.py` → `scripts/phase3.py::build_contract_features`. That last stage, which produces `prev_cap_pct`, lives in an experiment script. Stage 1 is verified faithful — all 14 model features reproduce at 100% on unchanged rows.
11. **Prorated partial-season rows are still in the training set** (~16%, see "Known contamination"). Removing them is a +0.0048 improvement waiting on a version bump.

## Reproducibility

```bash
# Verify the salary caps still reconcile — do this after touching config.py
python scripts/check_caps.py

# Rebuild the training data (three stages, in order)
python -c "import sys; sys.path.insert(0,'.'); from src.features.build_dataset import build_dataset; \
           build_dataset().to_csv('data/processed/training_data_v2.csv', index=False)"
python scripts/build_external_features.py
# stage 3: scripts/phase3.py::build_contract_features adds prev_cap_pct

# Train and evaluate
python src/model/train.py

# Full four-layer evaluation, champion vs challenger with a paired delta
python src/model/evaluate_suite.py

# Generate predictions (Grabit + CBA cap pipeline)
python src/model/predict.py

# Residual analysis, SHAP, signing-mechanism slices
python scripts/diagnostics.py
```

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*. — Censored-normal likelihood as XGBoost custom objective.
