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
| Basketball Reference | Salary, age, games played | 4,397 player-seasons, 2019–2030 |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage rates | 3,880 player-seasons |
| Spotrac | Signing mechanisms (Bird Rights, MLE, cap space, etc.) | 981 free agent signings |
| Manual curation | Awards (MVP, All-NBA, DPOY), draft position, salary cap history | Hand-verified CSVs |

**Target variable**: `cap_pct = annual_salary / salary_cap` (range 0–0.35). Normalizing to cap percentage enables cross-season comparison as the cap inflates ~5% annually.

**Intentional lag**: Performance metrics from season *t* predict salary in season *t+1*, reflecting how teams actually sign contracts — based on past production.

After filtering to year-1 contracts only (removing CBA-mandated escalator years) and slotted rookie-scale deals, the training set contains **1,487 player-season observations** across 789 unique players.

## Methodology

### Feature Engineering (14 features)

| Category | Features | Rationale |
|----------|----------|-----------|
| Performance | DARKO DPM, LEBRON, RAPM (all z-scored within season) | Three independent impact metrics capture value from different angles |
| Workload | minutes/game, usage rate, 3-year weighted availability | Playing time and durability signal |
| Demographics | age, age², height, draft pick | Age curve is nonlinear; draft pedigree carries a reputation premium |
| Context | cumulative award score, assist%, CBA era flag, previous contract value | Reputation, role, regime change, and anchoring effects |

Over 20 additional candidate features were tested and rejected via ablation (each evaluated by ΔCV R² with 10-seed averaging). Rejected features include team-level variables (win%, cap space), playoff performance (too sparse), and agent portfolio effects (data leakage when computed naively).

### Model: Two-Stage Grabit Pipeline

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

**Stage 2 — CBA Cap**

```
predicted_salary = min(latent_value, max_eligible_pct)
```

Max-eligible percentage differ from player to player and is calculated with respect to the actual CBA rule.

### Evaluation

5-fold GroupKFold cross-validation (same player never appears in both train and test) with 10-seed averaging for stability.

| Metric | Baseline XGBoost | Grabit + CBA Cap |
|--------|-----------------|------------------|
| CV R² (all years, 2019–2026) | 0.750 | **0.762** |
| CV R² (2024–2026 subset, n=385) | 0.839 | **0.852** |

The 2024–2026 subset R² better reflects forward prediction ability, as there is a major CBA rule change on 2023. The addition of Grabit is not a huge improvement due to the small number of contract it is affecting. However, I look forward to see it creating larger impact as it is applied to other contract limited by an external factor.

### Diagnostic Findings

Residual analysis by Spotrac signing type reveals systematic patterns:

- **Bird Rights / Sign & Trade**: model underpredicts by $2–3M — these mechanisms allow teams to exceed the cap for their own players, creating a retention premium invisible to the model
- **MLE / Minimum contracts**: model overpredicts by $1–2M — these players are often better than their salary, but are constrained by exception-level ceilings the model doesn't see
- **Cap Space signings**: roughly calibrated (bias < $1M)

These patterns confirm that CBA signing mechanisms create predictable distortions beyond what performance-based features can capture.

## Technical Contributions

1. **First application of Grabit (censored GBT) to sports salary prediction**: adapting a framework from credit-risk modeling to handle CBA-imposed salary ceilings
2. **Gated censoring**: Not all max-salary players are underpaid, and naively censoring them degrades predictions
3. **Systematic feature ablation**: 20+ features evaluated and documented with ΔCV R², preventing feature bloat common in sports analytics models


## Codebase

- **~6,300 lines** of Python across 47 files
- End-to-end pipeline: scraping → feature engineering → model training → prediction → diagnostics
- Fully reproducible: `python src/model/train.py` trains and evaluates from raw data

## Future Directions

- **Lower-bound censoring**: extend the framework to model veteran minimum and MLE ceilings (currently only max contracts are censored)
- **Temporal dynamics**: player trajectory modeling (growth curves for young players, decline curves for veterans) 
- **Agent / team negotiation features**: agent portfolio effects and team cap situation as interaction terms. I remember reading a paper which uses GNN for this.

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*, 106, 316–329.
- Hollinger, J. (2005). "BORD$: Basketball's Only Rational Dollar System." *Pro Basketball Prospectus*.
