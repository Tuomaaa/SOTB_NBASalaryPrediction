# Methodology — NBA Free Agent Valuation Model

## Objective

Predict a player's market value as **cap_pct** (annual salary / salary cap), using prior-season performance data. Inspired by Hollinger's BORD$, but fully open-source and reproducible.

## Data Sources

| Source | Data | Rows |
|--------|------|------|
| Basketball Reference | Per-season salary, age, team | 4,723 player-seasons (1,086 players, 2019–2031) |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage, box-score rates | 3,880 player-seasons (903 players) |
| Basketball Reference | Height in inches | 5,416 players |
| Basketball Reference | Pre-window salaries for the ceiling floor | 368 player-seasons (155 players, 2016–2018) |
| Spotrac | Signing mechanism, contract years, total value, AAV | 8,910 contract-seasons; 1,073 FA signings (2019–2026) |
| Manual reference (raw_external/) | Awards, draft position, team franchise value | Hand-curated CSVs |
| Manual reference | Salary cap by year, CBA parameters | config.py |

The salary table was refreshed on 2026-07-22, after 2026-27 free agency. The
previous scrape predated it, so the 2026 season held only players already under
contract — see "Cap values are load-bearing" and "Sample composition" below.
Spotrac labels were rebuilt on 2026-07-23; the 2016–2018 salaries were parsed
offline from the existing HTML cache, not scraped.

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
| `prev_cap_pct` | Year-1 cap_pct of the player's previous contract. Captures anchoring effect — prior contract value predicts next contract. First contracts are filled from the rookie scale by draft slot; see below. |

#### Filling `prev_cap_pct` for first contracts (v7.3x)

A first contract has no previous deal to observe. Until v7.3x the fill was one
number for everyone — the median `cap_pct` over first-round rows, which lands on
pick-4 money (~6.9% of cap). An undrafted player's actual predecessor is a
minimum deal (~1.5%), so the fill overstated it 4.6×, and because it was a
dataset-wide median it moved whenever rows were added (the 2026 refresh shifted
it 0.0704 → 0.0686, churning 1,769 rows).

The fill is now the expected rookie-scale value for the player's own draft slot:
per-pick medians taken over the dataset's own rookie-scale rows, interpolated
across gaps and forced monotone decreasing in pick, with minimum-level money past
pick 30. Derived from the data rather than the CBA salary tables, so it needs no
maintenance, and stable under refreshes because the rookie scale itself is.

Paired gain **+0.0034 ± 0.0007 (t = +4.91)**, positive in all five folds — three
times the Grabit effect, and the largest single paired improvement since the
two-stage pipeline.

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
| Post-hoc recalibration (linear) | -0.0013 | Nested, fold-honest. The model is already calibrated (slope 0.9999) |
| Post-hoc recalibration (isotonic) | -0.0052 | As above, and more prone to overfitting the fold |

### Training-set choices tested and rejected

| Change | ΔCV R² | Verdict |
|--------|--------|---------|
| Drop the 2020 (COVID) season from training | **-0.0037** (t = -1.21) | **Rejected.** The frozen cap and shortened season make 2020 look like a regime break, but removing its 167 rows hurt even the rows it was meant to protect: non-2020 rows fell 0.7686 → 0.7650. The season carries real pricing signal; it stays in training and is flagged in per-season tables. |

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

**Rose Rule**: players with ≤6 years experience who made an All-NBA team, won MVP,
or won DPOY qualify for 30% max. The award is looked up at the salary's own season
and at `s-1`, because a deal signed the summer after an All-NBA season starts the
year *after* the award.

**Supermax**: players with 7–9 years experience who earned elite awards qualify for
35% max. The lookback runs to `s-3`: designated-veteran deals can be signed two
summers before they take effect (Wall's came from All-NBA 2016-17, signed 2017,
effective 2019).

**No-decrease floor**: a veteran's first-year max is the *greater* of his tier and
105% of his previous salary, and legal raises cap season-over-season growth at 8%.
The ceiling therefore also carries a floor of `1.08 × previous season's cap_pct`.
Anchoring it needs pay from before the training window, so
`scripts/backfill_prehistory_salaries.py` re-parses the cached BBRef player pages
offline into `salaries_prehistory.csv` (368 rows, 155 players, 2016–2018).

#### Why the ceiling is audited

A ceiling below observed pay is provably wrong — the salary happened, so it was
legal — and it does damage twice: the row is mislabelled as censored at a
threshold it has already passed, and the Stage-2 clip pins the prediction below
the truth, a guaranteed error. Auditing `actual > max_eligible_pct` found 13 such
rows before v7.4x (Curry 2019 at 36.9% of a frozen cap against a 35% ceiling,
Wall and Towns on early-signed designated-veteran deals, and two rows that were
float dust at exactly the tier). All three rules above came from that audit; the
count is now **zero**, and it is worth re-running after any change to experience,
awards, or cap data.

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
slope of actual on predicted of 0.9999. Fold-honest recalibration confirms it by
failing to help (linear −0.0013, isotonic −0.0052).

### D — guards

- **Fixed evaluation set** whenever the training filter changes. R²'s denominator moves with the row set, so R² across different datasets is not comparable. The apparent v3.5x → v4.0 collapse from 0.866 to 0.645 is this effect, not a regression.
- **Baseline ladder**, so absolute R² is not mistaken for skill: predicting the mean 0.000, `mpg` alone **0.5740**, `mpg + prev_cap_pct` 0.6179, four features 0.7456, full 14 features 0.7609. Minutes per game alone reaches 75% of the full model's R²; the remaining ten features together buy +0.015 over that four-feature model.
- **Locked confirmation split**: 15% of players by stable hash, held out of every selection decision, opened at a version bump. Currently 0.7415 (n=186) against 0.7643 on the selection pool. The two levels are not comparable to each other — different players, different difficulty — so only the *trend* is informative, and it has widened from +0.0150 at v7.2x to +0.0228 at v7.4x. See "Watching the confirmation split" below.

### Results (v7.4x, 10 seeds, n = 1,297)

| Metric | Baseline XGBoost | Grabit + CBA cap |
|--------|-----------------|------------------|
| A1 CV R² | 0.7609 | 0.7609 |
| A2 CV R² (2024-26, n=397) | 0.8348 | **0.8366** |
| B1 forward R² (2024-26) | 0.8213 | **0.8240** |
| B1 95% CI | [0.769, 0.863] | [0.771, 0.867] |
| CV MAE | $3.32M | **$3.30M** |
| Calibration slope | 1.0163 | **0.9999** |
| Spearman | 0.8067 | 0.8062 |

Forward R² by origin (Grabit): 2024 → 0.844 (n=134), 2025 → 0.798 (n=139),
2026 → 0.827 (n=124). The cost of the forecasting setup relative to A2 is
−0.0126.

### Judging Grabit: the zone scorecard

**Pooled paired delta (Grabit − baseline): −0.0001 ± 0.0010, t = −0.07.** At
v7.2x this read +0.0011 (t = +2.72); part of that advantage was the ceiling bug
corrected in v7.4x, where clipping toward too-low ceilings happened to land on
the mislabelled rows it was clipping onto.

The pooled test is nonetheless the wrong instrument. Gated censoring touches 65
of 1,297 rows — 5% — so a pooled statistic divides the effect by twenty and
mistakes dilution for weakness. The suite reports a **zone scorecard** over the
rows Grabit exists for, those paid ≥90% of their own ceiling:

| Grabit zone (n=65) | Baseline | Grabit |
|---|---|---|
| MAE | $7.27M | **$6.52M** |
| bias | −$6.85M | −$6.43M |
| rows better / worse | — | **59 / 6** |

Bias stays near −$6M in this zone by construction, not by failure: these players
are paid their ceiling and the Stage-2 clip caps predictions at that ceiling, so
the residual can only be ≤ 0. MAE is the number that moves.

Spillover is real and two-sided, because one set of trees serves every row. Rows
at 75–90% of their ceiling are not censored yet gain (bias −$3.21M → −$2.74M);
rows at 50–75% pay for it (MAE +$0.13M, 67 rows worse against 39 better). The net
is clearly positive, and no gating threshold can separate the two — they are the
same mechanism.

### Acceptance rules

**Challenger changes** — features, filters, hyperparameters, anything acting on
every row — are accepted when all four hold:

1. paired A1 delta > 0 with |Δ| / SE > 2
2. A2 moves the same direction (significance not required)
3. no C2 segment's bias worsens by more than $0.3M
4. MAE does not regress beyond the agreed tolerance

**Grabit** is judged on its zone alone: keep it while the zone MAE delta is
negative, drop it when the zone itself turns positive. Applying the pooled rule
to a 5% intervention would have removed it at v7.4x on a t-statistic of −0.07.

### Watching the confirmation split

Every accept/reject in this project's history reads the same metric. Each decision
carries noise and the winning side is kept, so across 20+ feature decisions, two
hyperparameter sweeps, and the σ and gate thresholds, the metric can drift upward
without the model improving. The confirmation split is the canary: it carries only
15% weight in the decision metric, so it should climb more slowly than the
selection pool — but not fall while the pool rises.

| | selection pool (n=1,111) | confirmation (n=186) | gap |
|---|---|---|---|
| v7.2x | 0.7611 | 0.7461 | +0.0150 |
| v7.4x | 0.7643 | 0.7415 | +0.0228 |

Three reasons not to act yet: an R² on 186 rows has a bootstrap interval about
±0.05 wide, so a −0.005 move is well inside noise; and neither recent change looks
like metric-mining — v7.3x was a semantic correction motivated independently of
the metric, and v7.4x *lowered* pooled R² in exchange for correctness, which is
the opposite of what mining produces.

Two honest caveats about the guard as built. The decision metric still *includes*
the confirmation rows, so it is a diluted version of the real thing; and although
the protocol says "opened at a version bump", the suite prints it on every run.
It is a canary, not a sealed envelope. At the next version bump, re-score each
accepted change on confirmation rows only — the per-row OOF for every version is
archived in `outputs/models/oof_reference.csv`. If accepted changes are
systematically ≤0 there while >0 on the selection pool, tighten the protocol so
selection metrics are computed on the selection pool alone.

## Holdout vs Valuation

Two distinct scoring modes, and CONTEXT.md gives them separate names because
they mean different things:

- **Signing Board / holdout**: Year-1 filter on train and test both. Produces the **Signing Residual** — a measure of model accuracy against a price the market actually set.
- **Value Board / valuation**: Year-1 filter on train only, score every row. Produces the **Contract Surplus** — a statement about a team's books, not about model error. R² is not meaningful here; ranking is.

## Diagnostic Findings (v7.4x)

### Bias by predicted band

| Predicted band | n | bias | MAE |
|---------------|---|------|-----|
| <2% | 211 | +$0.00M | $0.49M |
| 2-4% | 348 | +$0.13M | $2.17M |
| 4-8% | 344 | +$0.59M | $3.97M |
| 8-15% | 213 | −$0.27M | $5.04M |
| 15-25% | 124 | −$1.03M | $5.71M |
| 25%+ | 57 | +$1.44M | $4.77M |

Flat to within ±$1.05M, with the widest cells at the two thinnest bands.

### Signing Residual by mechanism

| Mechanism | n | bias | MAE |
|-----------|---|------|-----|
| Bird Rights | 353 | −$2.43M | $4.82M |
| Minimum | 309 | +$2.51M | $2.59M |
| MLE | 204 | +$1.32M | $2.26M |
| Cap Space | 143 | −$0.78M | $3.61M |
| Unknown | 132 | +$0.59M | $2.66M |
| Early Bird | 57 | −$1.05M | $2.69M |
| Other | 49 | +$1.68M | $2.14M |
| Non-Bird | 27 | +$2.07M | $2.58M |
| Sign & Trade | 20 | −$4.62M | $5.40M |

Bird Rights and Sign & Trade are underpriced — both let a team exceed the cap for
its own player, and the retention premium is invisible to the features. Minimum
and MLE are overpriced. These patterns survive controlling for predicted value
but are still not usable as features; see "Why signing mechanism cannot be a
feature."

**Label coverage was rebuilt on 2026-07-23** and now runs 85–91% of year-1
evaluation rows per season, against ~43% before. `Unknown` fell from 57% of the
frame to 10%. Three parsing faults were fixed in
`scripts/scrape_spotrac_players.py` and the pipeline is now
`scripts/refresh_spotrac.py`:

- Season assignment walked the career-earnings table backwards, so one skipped
  deal shifted every assignment below it — a $27M season carried a "Minimum"
  label. Each contract now anchors itself at `fa_year − n … fa_year − 1` from its
  own Free Agent field.
- A mid-season buyout puts two real contracts on one season (Westbrook 2022:
  supermax cash, then a minimum signing). Overlapping spans are kept and the
  consumer picks the contract whose AAV is closest to the row's observed salary,
  since the label a residual diagnostic needs is the contract that *produced*
  the salary.
- Cache slugs drop dots, so round-tripping "a.j. green" through a slug silently
  unlabelled every dotted or hyphenated name. Slugs now map back through the
  training data.

Two consequences for reading the table above. `Bird Rights` jumped from n=58 to
n=353 because extensions now resolve and bucket there — it conflates re-signings
with extensions, so do not read it as a clean retention-premium estimate without
splitting them first. And Minimum-labelled rows above $6M fell from 30 to 6;
`ISSUES.md` #5 lists the residual gaps.

### Grabit impact in its zone

See "Judging Grabit: the zone scorecard" above. Over the 65 rows paid ≥90% of
their own ceiling, MAE falls $7.27M → $6.52M with 59 rows better and 6 worse.

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
3. **Signing mechanism is a label, not a feature** — partly determined by the contract itself, and empirically worthless as a modelled probability. See the dedicated section above.
4. **No tracking data**: NBA.com tracking data (drives, catch-and-shoot, rim protection) could improve archetype-specific predictions.
5. **σ tension**: CV prefers σ=0.02 (conservative censored gradients), holdout prefers larger σ (more aggressive). Sticking with CV-optimal to avoid overfitting to a single holdout season.
6. **Point estimates against a bimodal target**: `y | x` is genuinely bimodal for mid-market players. A conditional mean is the R²-optimal point estimate and is nonetheless wrong for both modes. A distributional output — quantiles, or `P(signs for the minimum)` alongside a conditional market value — would answer the actual question better, and would not be measurable by R².
7. **The model is not underfitting**: depth 6, learning rate 0.03, and looser `min_child_weight` all score worse than the current settings (0.7648 / 0.7613 / 0.7527 against 0.7661). Added structure cannot be justified as an inductive-bias fix.
8. **Grabit's reach is not confined to its gate.** Censoring changes gradients for 65 rows, but one set of trees serves all 1,297, so leaf values shift for their neighbours too. Rows at 75–90% of their ceiling gain without being censored; rows at 50–75% lose slightly. No threshold separates the two — it is the same mechanism, and the net is positive.

### Pipeline and reproducibility

9. **The contract-structure script was never committed.** Only `contract_structure_v2.csv` survives. Reconstructing the detection from CBA escalator ratios reaches at best 88% agreement on the year-1 flag and 74% on contract length across a 30-point parameter sweep — far too low to regenerate history without invalidating every published version number. `scripts/extend_contract_structure.py` therefore extends the table incrementally: rows whose salary is unchanged keep their assignment byte-for-byte, only new or changed rows are assigned, and it hard-fails if an unchanged row would move. Any future change needing a full recompute will still hit this wall.
10. **`prev_cap_pct` is produced by an experiment script.** The regeneration chain is `src/features/build_dataset.py` → `scripts/build_external_features.py` → `scripts/phase3.py::build_contract_features`, and that last stage lives in `phase3.py` for historical reasons. `scripts/rebuild_training_data.py` now chains all three behind one command and validates that the fifteen cap-independent columns reproduce exactly on shared rows, but the stage itself has not been moved to a home of its own.

## Reproducibility

```bash
# Verify the salary caps still reconcile — do this after touching config.py
python scripts/check_caps.py

# Refresh the data after a signing period. Run in this order: the salary step
# snapshots the pre-refresh table for the structure step to diff against.
python scripts/refresh_salaries.py            # ~25 min live; --use-cache to re-parse
python scripts/extend_contract_structure.py
python scripts/rebuild_training_data.py       # chains all three build stages

# Signing-mechanism labels for a new FA year (diagnostic only, never a feature)
python scripts/refresh_spotrac.py --year 2026 --refresh-signees
python scripts/refresh_spotrac.py --reparse-only   # no network

# Pre-2019 salaries for the ceiling's no-decrease floor (offline, from cache)
python scripts/backfill_prehistory_salaries.py

# Train and evaluate
python src/model/train.py

# Full four-layer evaluation, champion vs challenger, paired delta + Grabit zone
python src/model/evaluate_suite.py

# Generate predictions (Grabit + CBA cap pipeline)
python src/model/predict.py

# Residual analysis, SHAP, signing-mechanism slices
python scripts/diagnostics.py
```

Every number quoted in this document can be reproduced by checking out its
version tag (`v7.1x` … `v7.4x`) and running `src/model/evaluate_suite.py`.

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*. — Censored-normal likelihood as XGBoost custom objective.
