# NBA Free Agent Valuation Model — Project Brief

## Motivation

NBA teams spend over 4billion annually on player salaries, yet no publicly available, reproducible model exists for estimating player market value. Hollinger's BORD$ (ESPN) remains the most cited valuation framework, but its methodology is proprietary and closed-source.

This project builds an open-source NBA player valuation model that predicts contract value as a percentage of the salary cap, using only publicly available data. It tackles a key econometric challenge unique to the NBA: **right-censoring from the Collective Bargaining Agreement (CBA)**, which imposes hard salary ceilings that suppress observed salaries for elite players.

## Glossary
* **Collective Bargaining Agreement (CBA)**: An agreement that imposes hard salary ceilings that suppress observed salaries for elite players.
* **Salary Cap**: A league-wide financial limit that inflates roughly 5% annually.
* **Max Contract / Max-eligible**: A rule capping a player's maximum salary at 25–35% of the salary cap, depending on their experience.
* **Albatross Contracts**: Contracts given to players who are extremely overpaid despite declining performance.
* **Escalator Years**: Subsequent years in a multi-year contract that do not reflect a player's actual value.
* **Rookie-scale Deals**: Pre-determined contracts specifically designed for newly drafted players.
* **Bird Rights / Sign & Trade**: Mechanisms that allow teams to exceed the salary cap to retain their own players, which creates a retention premium invisible to standard models.
* **MLE (Mid-Level Exception) / Minimum Contracts**: Signing exceptions with hard ceilings that often constrain players who actually perform better than their observed salary.
* **Cap Space Signings**: Standard free agent signings using available team budget, which reflect a relatively calibrated market value with minimal bias.
* **DARKO DPM, LEBRON, RAPM**: Three independent impact metrics used to capture a player's value from different angles.
* **Usage Rate**: A metric that signals a player's playing time and impact.
* **Awards (MVP, All-NBA, DPOY)**: Major league accolades (Most Valuable Player, All-NBA, Defensive Player of the Year) that act as context features reflecting a player's reputation and role.

## Research Question

> Given a player's prior-season performance, age, durability, and league context, what is their market value — and how do CBA constraints distort observed prices?

## Data

| Source | Content | Scale |
|--------|---------|-------|
| Basketball Reference | Salary, age, games played | 4,723 player-seasons, 2019–2031 |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage rates | 3,880 player-seasons |
| Spotrac | Signing mechanisms (Bird Rights, MLE, cap space, etc.) | 2,403 contract-seasons |
| Manual curation | Awards (MVP, All-NBA, DPOY), draft position, salary cap history | Hand-verified CSVs |

**Target variable**: `cap_pct = annual_salary / salary_cap` (range 0–0.35). Normalizing to cap percentage enables cross-season comparison as the cap inflates ~5% annually.

**Intentional lag**: Performance metrics from season *t* predict salary in season *t+1*, reflecting how teams actually sign contracts — based on past production.

After filtering to year-1 contracts only (removing CBA-mandated escalator years), slotted rookie-scale deals, prorated partial seasons, and rows a three-signal test identifies as escalator years mislabelled as fresh signings, the training set contains **1,172 player-season observations** across 581 unique players.

**A caveat about the denominator.** Because the salary cap divides the target, a
wrong cap rescales an entire season. Two seasons carried stale pre-media-deal
projections until they were caught by dividing max contracts back out: the
implied cap reproduces the configured value exactly for five seasons and
disagreed for 2025 and 2026, which had been inflating those targets by ~9%.
Correcting them moved the model's calibration slope to 1.0006. The check now
runs as an assertion over every season.

## Methodology

### Feature Engineering (15 features)

| Category | Features | Rationale |
|----------|----------|-----------|
| Performance | DARKO DPM, LEBRON, RAPM (all z-scored within season) | Three independent impact metrics capture value from different angles |
| Workload | minutes/game, usage rate, 3-year weighted availability | Playing time and durability signal |
| Demographics | age, age², height, draft pick | Age curve is nonlinear; draft pedigree carries a reputation premium |
| Context | cumulative award score, assist%, CBA era flag, previous contract value, recent prior waiver | Reputation, role, regime change, anchoring, and previous-contract termination context |

Over 20 additional candidate features were tested and rejected via ablation (each evaluated by ΔCV R² with 10-seed averaging). Rejected features include team-level variables (win%, cap space), playoff performance (too sparse), and agent portfolio effects (data leakage when computed naively).

### Model: Three-Stage Grabit Pipeline

**The core problem**: The CBA caps maximum salary at 25–35% of the cap depending on experience. A player worth 40% of the cap is paid 35% — their observed salary is a **right-censored** observation, not their true market value. Standard regression learns to predict the ceiling, permanently underpredicting elite players.

**Stage 1 — Grabit (Gradient Tree-Boosted Tobit)**

We adapt the Grabit framework (Sigrist & Hirnschall, 2019) to NBA salary prediction. XGBoost is trained with a custom censored-normal likelihood:

- **Uncensored rows** (non-max contracts): standard squared-error gradients
- **Censored rows** (max contracts): inverse Mills ratio gradients push latent predictions above the observed ceiling

```
grad_censored = -σ × φ(z) / Φ(z)    where z = (pred - observed) / σ
```

This produces a **latent value** — the player's unconstrained market worth.

**Gated censoring**: Not all max contracts represent ceiling-constrained players. "Albatross contracts" (e.g., A player who is extremely overpaid) are max-paid despite declining performance. We gate censoring by requiring the baseline XGBoost prediction to exceed 55% of the max-eligible salary, filtering ~5 such cases per training run.

**Stage 2 — CBA bounds (push + clip)**

Where a route classifier says P(max) >= 0.52, the latent is pushed a P-weighted
fraction of the way toward 1.05 times the tier ceiling. The clip then caps the
prediction at the player's Max-Eligible Percentage from above and lifts it to the
Floor Percentage from below. Composition lives in `src/model/stages.py`.

**Stage 3 — Extension raise cap**

A first-paying-year extension is additionally clipped at its legal raise cap:
120% (2017 CBA) or 140% (2023 CBA) of the player's prior salary, or the same
multiple of the league's Estimated Average Player Salary, whichever is greater.
Stage 3 requires knowing the signing route and is the only told-route component.

### Evaluation

Four layers, each answering a different question, because the obvious single
number is ambiguous between them.

**Selection** uses pooled 5-fold GroupKFold CV over 10 seeds, grouped so a
player never appears in both train and test. Training on a later season to score
an earlier one is legitimate here: the estimand is the market's pricing
function, a structural quantity rather than a forecast. Tested directly, time
direction is not distinguishable from noise at matched training-set size
(+0.029 ± 0.021, t = 1.36).

**Forecasting** uses rolling origins — train on every season before *T*, score
*T*, for *T* in 2024–2026. This is what the inference pipeline actually does.
Earlier origins are excluded because their training sets are a third to a fifth
of the current one, so they measure data scarcity rather than the model.

| Metric | Baseline XGBoost | Grabit, two-sided |
|--------|-----------------|------------------|
| CV R² (2019–2026, n=1,172) | 0.7633 | **0.7653** |
| CV R² (2024–2026, n=386) | — | **0.8465** |
| Forward R² (2024–2026) | — | **0.8324** [0.779, 0.877] |
| CV MAE | — | **$3.065M** |
| Calibration slope | — | 0.9885 |

Deltas are paired by fold because fold-to-fold variation (sd 0.044) is thirty
times seed-to-seed variation (sd 0.0014); comparing two independently-reported
means would discard nearly all the power. Since 2026-07-23 the paired delta that
*decides* is computed on the selection pool alone — an audit found accepted
changes helping watched rows while hurting the held-out 15% (difference-in-
differences +5.6e-5, cluster CI excluding zero), which is what fitting a metric
rather than a market looks like.

**Censoring is judged where it acts, not on the pooled average.** It touches 354
of 1,172 rows and the two sides pull opposite ways, so a pooled statistic
averages two targeted effects over rows neither touches:

| Zone | Baseline | Censored | rows better/worse |
|---|---|---|---|
| Max (n=57, paid ≥90% of own ceiling) | $7.07M MAE | **$6.11M** | 55 / 2 |
| Floor (n=297, pinned at the minimum) | $2.40M MAE | **$1.98M** | 254 / 43 |

The floor side also moves the pooled dollar metric — MAE $3.181M → $3.065M, 95%
cluster-bootstrap CI [−0.139, −0.091] — while barely touching R², because R²
weights by squared error and these are 297 small-dollar rows. Where the two
metrics disagree this sharply the disagreement is arithmetic, not contradiction.

**Integrity and guards** cover the ways this kind of model quietly misleads: a
baseline ladder (minutes per game *alone* reaches 0.587 against the full model's
0.758), a 15%-of-players confirmation split held out of every selection decision,
and a rule that any comparison across different training filters runs on a fixed
evaluation set, since R²'s denominator moves with the row set.

### Diagnostic Findings

**A correction worth stating plainly.** Earlier versions of this work reported
that the model underpaid stars by up to $6M, read off residuals binned by actual
salary. That gradient is an artifact — binning by a noisy target produces it even
for a perfectly calibrated model. Binned by *predicted* value instead, bias is
flat to within ±$0.51M, and the OLS slope of actual on predicted is 0.99.
Fold-honest recalibration confirms it by failing to help.

The same trap has a second form, met while comparing two models: cutting
segments on either model's *own* predictions lets band composition shift between
them and manufactures a difference that is not there. Model comparisons here
assign rows to segments by something neither model produced.

What survives the correction is the signing-mechanism pattern, which holds after
controlling for predicted value:

- **Bird Rights / Sign & Trade**: underpriced by ~$3M. Both let a team exceed the cap for its own player, and that retention premium is invisible to the features.
- **MLE / Minimum**: overpriced by ~$1.4M — players better than the exception-level ceiling they signed under.
- **Cap Space**: roughly calibrated (−$0.68M).

The tempting inference — model the mechanism, recover the error — does not work,
and establishing that is one of this project's more useful negative results.
Mechanism is itself a function of the player features, so a fold-honest
`P(mechanism | x)` *lowers* CV R² by 0.0073. A leaky oracle supplying the
realised mechanism gains only +0.0137. The residual gap is the spread of a
genuinely bimodal conditional distribution: given the same inputs a player
either lands a market deal or takes a minimum, and the conditional mean sits
correctly between the two modes while being wrong for both. Any real gain must
come from information that is *not* a function of the current features.

**One bound was recoverable, and the distinction is economic.** A *good* player
on a minimum could have signed elsewhere for more, so his salary reflects a
choice and censoring him is unjustified — that is the negative result above. A
player whose unconstrained price sits *below* the league minimum is held there
by a rule no contract may cross, exactly as a max player is held at the ceiling.
Left-censoring that population is worth $0.42M per row across 297 rows, with a
bootstrap interval clear of zero.

## Technical Contributions

1. **First application of Grabit (censored GBT) to sports salary prediction**: adapting a framework from credit-risk modeling to handle CBA-imposed salary bounds, censored on both sides — ceilings for max contracts, floors for veteran minimums
2. **Gated censoring**: Not all max-salary players are underpaid, and naively censoring them degrades predictions
3. **Systematic feature ablation**: 20+ features evaluated and documented with ΔCV R², preventing feature bloat common in sports analytics models
4. **A characterised negative result**: signing mechanism produces a large, real residual pattern that is nonetheless unrecoverable, because the mechanism is a function of the same features and the conditional distribution is bimodal. The upper bound is measured, not assumed
5. **An evaluation protocol that separates what the single headline number conflates**: structural estimation from forecasting, calibration from accuracy, and selection from confirmation — with the failure modes it exists to prevent documented as worked examples

## Codebase

- **~6,700 lines** of Python across 42 files
- End-to-end pipeline: scraping → feature engineering → model training → evaluation → prediction → diagnostics
- `python src/model/train.py` trains and evaluates; `python src/model/evaluate_suite.py` runs the full four-layer protocol with a paired champion/challenger comparison
- Two gaps in reproducibility are documented rather than papered over: the contract-structure detection script was lost before it was committed, and regenerating the training data spans three files with no chaining entry point

## Future Directions

- **Distributional output instead of a point estimate.** Since `y | x` is demonstrably bimodal, a conditional mean is structurally the wrong deliverable — it is wrong for both modes by construction. Predicting quantiles, or `P(signs for the minimum)` alongside a conditional market value, answers the question the project actually asks: what a player is *worth*, not what he will be *paid*. This is also the coherent home for the low-end censoring idea, which cannot work as a point-accuracy fix.
- **Target the whole contract, not year one.** Contract length is already in the data. "Market value" is the full deal; year-1 salary carries arbitrary front- and back-loading as noise.
- **Information orthogonal to the current features** — market supply and demand at the position that summer, league-wide cap room, how the previous contract terminated. The oracle bound says this is the only route to a real accuracy gain.
- **Temporal dynamics**: player trajectory modeling (growth curves for young players, decline curves for veterans).

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*, 106, 316–329.
- Hollinger, J. (2005). "BORD$: Basketball's Only Rational Dollar System." *Pro Basketball Prospectus*.
