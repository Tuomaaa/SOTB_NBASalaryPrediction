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

## Phase 6: Two-stage pipeline (Grabit + CBA cap)

| Ver | Model | 10-seed CV R² | 10-seed Holdout R² | N | Feat | Change |
|-----|-------|---------------|-------------------|---|------|--------|
| 7.0x | XGBoost (Grabit) | **0.7612 ± 0.0012** | **0.8382 ± 0.0040** | 1487 | 14 | Two-stage: nonlinear Tobit (censored loss) + CBA cap. σ=0.02, gated censoring |

v7.0x pipeline:
- **Stage 1 (Grabit)**: XGBoost with custom censored-normal loss. Max-contract players (cap_pct ≥ 90% × max_eligible) are right-censored — their observed salary is a ceiling, not true value. The censored loss pushes latent predictions past the ceiling to estimate unconstrained market value.
- **Stage 2 (CBA cap)**: `min(latent, max_eligible_pct)` clips predictions to CBA salary limits.
- **Gated censoring**: only censor max rows where baseline prediction ≥ 55% of max_eligible (filters albatross contracts like John Wall, Gordon Hayward).
- **Rose Rule / Supermax detection**: non-lagged elite award lookup (All-NBA, MVP, DPOY) determines max_eligible tiers — 25% (0-6 yrs), 30% (Rose Rule or 7-9 yrs), 35% (10+ yrs or Supermax).

Experiments leading to v7.0x:
- Simple CBA cap on baseline XGB: CV -0.0008. XGB already learns ceiling; problem is underprediction, not overprediction.
- Linear Tobit: CV 0.6692. Far too weak — linear model can't capture non-linearities.
- Hybrid (XGB + linear Tobit): best variant CV 0.7605. Only helped 35% tier, 30% tier untouched.
- Grabit V1 (all max censored): CV 0.7591. All three max tiers improved but 25% tier regressed due to Rose Rule lag bug.
- Grabit V2 (gated censoring): CV 0.7594. Filters albatross contracts.
- **Grabit V3 (fixed Rose Rule)**: CV 0.7612. Non-lagged elite award detection fixes Edwards/Haliburton/Trae/Cade tier assignment (25%→30%).

Signing type breakdown (OOF, $M):

| Type | N | Bias (v6.2x) | Bias (v7.0x) | MAE (v6.2x) | MAE (v7.0x) |
|------|---|-------------|-------------|------------|------------|
| Bird Rights | 58 | -$2.6M | -$2.7M | $4.9M | $4.9M |
| Sign & Trade | 17 | -$2.6M | -$2.3M | $6.2M | $6.1M |
| Cap Space | 51 | -$0.8M | -$0.8M | $4.2M | $4.2M |
| Early Bird | 27 | -$0.4M | -$0.4M | $2.6M | $2.7M |
| MLE | 61 | +$1.4M | +$1.4M | $2.5M | $2.5M |
| Minimum | 419 | +$1.5M | +$1.5M | $2.4M | $2.4M |
| **Total** | **1487** | **-$0.0M** | **+$0.0M** | **$3.2M** | **$3.1M** |

## Phase 7: Data correction + evaluation protocol

| Ver | Model | 10-seed CV R² | Forward R² (24-26) | N | Feat | Change |
|-----|-------|---------------|-------------------|---|------|--------|
| 7.1x | XGBoost (Grabit) | 0.7581 | 0.8249 | 1556 | 14 | 2026 signings refreshed, 2025/26 caps corrected, four-layer evaluation suite |
| 7.2x | XGBoost (Grabit) | 0.7588 | 0.8216 | 1297 | 14 | Prorated partial-season salaries removed from training |
| 7.3x | XGBoost (Grabit) | 0.7622 | 0.8237 | 1297 | 14 | First-contract `prev_cap_pct` filled from the rookie scale by draft slot. 2020 season tested and kept |
| 7.4x | XGBoost (Grabit) | 0.7609 | 0.8240 | 1297 | 14 | Stage-2 ceiling audit — no-decrease rule, supermax `s-3` lookback, prehistory backfill. Over-cap rows 13 → 0 |
| 7.5x | XGBoost (Grabit) | 0.7610 | 0.8239 | 1297 | 14 | Curated early-supermax list replaces the blanket `s-3` lookback, which had un-censored two genuine max signings |
| 7.6x | XGBoost (Grabit) | 0.7635 | 0.8200 | 1291 | 14 | Year-1 rows paid above their *tier* ceiling demoted — 6 provable escalator mislabels |
| 7.7x | XGBoost (Grabit) | 0.7647 | 0.8283 | 1172 | 14 | Three-signal continuation demotion — 119 stale escalator rows out. Largest paired gain of the phase |
| 7.8x | XGBoost (Grabit v4) | 0.7653 | 0.8324 | 1172 | 14 | Two-sided Grabit — left-censoring at the CBA floor. Pooled MAE $3.18M → $3.07M |
| 7.9x | XGBoost (Grabit v4) | 0.7631 | 0.8191 | 949 | 14 | Continuation filter v2 — dated-span demotion. Frame 1,172 → 949 (342 demoted, 0 contradicted) |
| 7.10x | XGBoost (Grabit v4) | 0.7849 | 0.8276 | 944 | 14 | `prev_cap_pct` repair + 2026 cap correction. Paired ΔSel +0.0135 (t 7.6). Frame 949 → 944 |
| 7.11x | XGBoost (Grabit v4) | 0.7856 | 0.8298 | 944 | 14 | `prev_cap_pct` = previous-season pay (feature-batch Arm C, boundary call). ΔSel +0.00178 (t 1.96) |
| 7.12x | XGBoost (Grabit v4) | 0.7878 | 0.8303 | 944 | 14 | Designated-ceiling award-path fix. Max zone 56 → 68, zone MAE 6.18 → 4.44 |
| **7.13x** | **XGBoost (Grabit v4)** | **0.7865** | **0.8342** | **944** | **14** | **Real service years + correctness debt. Max zone 68 → 70; awards name-join repair (83 dirty rows); Kanter alias; canonical 10-seed CV; `predict.py` rewired to champion stack** |

### Versioning convention

Every change that moves a published number — data, target, features, model, or
evaluation protocol — takes the next `vN.Mx` and is tagged on its landing commit
with the headline A1/A2/B1 in the tag message, so any quoted figure can be
reproduced with a checkout. Bug fixes count (v5.3x was one); the *kind* of change
belongs in the Change column, not the numbering. Diagnostics, tooling and
documentation changes do not consume a number.

Tags: `v7.1x` `09dbe4e` · `v7.2x` `cb27319` · `v7.3x` `b31a6fb` · `v7.4x` `ecdf3da` ·
`v7.5x` `c17880d` · `v7.6x` `d7c72ef` · `v7.7x` `63e290c` · `v7.8x` `c03eb9a` ·
`v7.9x` `ddf4bae` · `v7.10x` `c56116b` · `v7.11x` `643124f` ·
`v7.12x` `61bdfcd` · `v7.13x` `981e6dc`.

### v7.1x: refreshed data and corrected caps

**The headline numbers fell and that is the correct outcome.** Three separate
corrections, none of them a model change:

1. **The 2026 test set was optimistically biased.** The BBRef contract cache was
   scraped 2026-06-26, days before 2026-27 free agency opened, so the 2026 season
   held only extensions and players already under contract — the easy cases.
   Refreshing all 30 team pages took usable 2026 rows from 55 to 124 and dropped
   the 2026 forward R² from 0.843 to 0.811 on a test set that finally represents
   the task.
2. **The 2025 and 2026 salary caps were wrong**, sitting at stale
   pre-media-deal projections ($141.208M and $153M against $154.647M and $166M),
   inflating those seasons' targets by ~9%. Correcting them moved the calibration
   slope from 1.014 to 1.0006.
3. **The reported uncertainty was the wrong quantity.** The `± 0.0009` figures in
   Phases 5-6 are seed-averaging noise. Fold-to-fold sd is 0.045 — forty-five times
   larger. Those comparisons remain valid because they were paired on identical
   folds, but the intervals understated the uncertainty on any single R².

v7.1x is therefore **not** comparable to v7.0x and earlier: different rows,
different target values for two seasons. It is the first entry measured under
the protocol in `src/model/evaluate_suite.py`.

### v7.2x: prorated salaries removed

259 of the 1,556 training rows (17%) paid less than 1.2% of the cap, a median of
$0.29M against a veteran minimum that never drops below 1.30% of the cap in any
season covered. They are 10-day contracts and mid-season signings: the same
player at the same ability lands on a different target depending only on the date
he signed. `_filter_prorated` in `train.py` now cuts them, using a share of the
cap rather than a dollar figure so the floor tracks the minimum scale.

**This is a calibration fix, not an accuracy gain.** Including those rows drags
every prediction down by roughly half a million dollars. Scoring the same Grabit
champion on the same 1,297 clean rows, varying only the training set:

| trained on | CV R² | MAE | bias |
|---|---|---|---|
| all rows (v7.1x) | 0.7534 | $3.289M | **−$0.423M** |
| prorated removed (v7.2x) | **0.7588** | $3.327M | **+$0.101M** |

The R² gain of +0.0054 is almost entirely the intercept: re-centring the v7.2x
predictions by their mean shift returns R² to 0.7522, within noise of the
unfiltered model. The conditional pricing function did not change — the level
did. Bias falls monotonically as the floor rises (−$0.481M at no filter,
−$0.075M at 0.6%, −$0.038M at 0.8%, −$0.014M at 1.0%, +$0.029M at 1.2%).

Headline movements are a **re-baseline, not a comparison** — the row set changed,
so R²'s denominator did too:

| Layer | R² 7.1x | R² 7.2x | MAE 7.1x | MAE 7.2x |
|---|---|---|---|---|
| A1 pooled CV | 0.7581 | 0.7588 | $3.14M | $3.33M |
| A2 2024-26 | 0.8257 | 0.8343 | $3.30M | $3.38M |
| B1 forward | 0.8249 | 0.8216 | $3.30M | $3.57M |

MAE rises across the board because the removed rows were the easiest in the set —
the old model predicted them to $2.40M MAE against $3.29M for real contracts, so
keeping them in the *evaluation* set was flattering the average. On identical
rows MAE moves only +$0.038M.

**The C2 segment guard fired and was overruled.** Minimum (+$0.84M → +$1.51M) and
MLE (+$1.05M → +$1.44M) both worsened past the $0.30M threshold. But every
segment's signed bias moved up by the same +$0.39M to +$0.72M — Bird Rights
−$3.06M → −$2.34M and Cap Space −$0.86M → −$0.29M improved by the identical
mechanism. The guard compares |bias|, so a global level correction necessarily
trips it wherever a segment was already overpredicted. **The guard needs to
measure segment bias relative to the global level; until it does, treat a C2
breach on a calibration change as uninformative.**

### v7.3x: rookie-scale fill for first contracts

`prev_cap_pct` anchors a player's next deal on his previous one, but a first
contract has no previous deal to observe. The old fill was a single number for
everyone — the median `cap_pct` over first-round rows, which lands on pick-4
money (~6.9% of cap). An undrafted player's actual predecessor is a minimum deal
(~1.5%), so the fill overstated it 4.6×; and being a dataset-wide median it
drifted on every refresh, churning 1,769 rows on the 2026 one.

The fill is now the expected rookie-scale value for the player's own draft slot —
per-pick medians over the dataset's own rookie-scale rows, interpolated and
forced monotone in pick — with minimum-level money past pick 30. Derived from the
data rather than the CBA tables, so it needs no maintenance and is stable under
refreshes because the rookie scale itself is.

| Layer | v7.2x | v7.3x | paired Δ |
|---|---|---|---|
| A1 pooled CV | 0.7588 | **0.7622** | **+0.0034 ± 0.0007, t = +4.91** (all five folds positive) |
| A2 2024-26 | 0.8343 | 0.8370 | same direction |
| B1 forward | 0.8216 | 0.8237 | same direction |

C2's worst |bias| growth was +$0.07M against the $0.30M gate. At three times the
Grabit effect this is the largest single paired improvement since the two-stage
pipeline itself — from a semantic correction, not a search over the metric.

**The 2020 season was tested in the same pass and kept.** Training *without* 2020
scored a paired −0.0037 (t = −1.21) and hurt even the rows it was supposed to
protect: non-2020 rows fell 0.7686 → 0.7650. The COVID season's 167 rows carry
real pricing signal despite the frozen cap. No code change; the open question is
closed.

### v7.4x: Stage-2 ceiling audit

A ceiling below observed pay is provably wrong — the salary happened, so it was
legal — and it does damage twice: the row is mislabelled as censored at a
threshold it has already passed, and the Stage-2 clip pins the prediction under
the truth, a guaranteed error. Auditing `actual > max_eligible_pct` found 13 such
rows in three classes, all now closed.

1. **Missing no-decrease rule, plus a data boundary.** A veteran's first-year max
   is the greater of his tier and 105% of prior pay, and legal raises cap
   season-over-season growth at 8%, so the ceiling gains a floor of 1.08 × the
   previous season's pay. The 2019 rows had no previous season to read — the data
   starts there — but the cached BBRef player pages carry full salary history, so
   `scripts/backfill_prehistory_salaries.py` re-parses them offline into
   `salaries_prehistory.csv` (368 rows, 155 players, 2016–2018). Fixes Curry 2019
   at 36.9% of a frozen cap against a 35% ceiling, plus Westbrook, Paul, Wiggins,
   McCollum.
2. **The qualifying award can precede the start season.** Designated-veteran deals
   sign up to two summers early — Wall's supermax came from All-NBA 2016-17,
   signed 2017, effective 2019. The supermax lookup now accepts an elite award
   back to `s-3` and the Rose path accepts `s-1` (Jaylen Brown and Towns 2024).
3. **Two rows were float dust** at exactly the tier (Giannis, Adebayo at
   35.00/25.00). The audit uses a $17K tolerance; no code changed.

| Layer | v7.3x | v7.4x |
|---|---|---|
| A1 pooled CV | 0.7622 | 0.7609 |
| A2 2024-26 | 0.8370 | 0.8366 |
| B1 forward | 0.8237 | 0.8240 |
| Calibration slope | 1.001 | **0.9999** |
| Over-cap rows | 13 | **0** |

**Part of Grabit's measured advantage was this bug.** With honest ceilings the
pooled paired delta against baseline XGBoost is −0.0001 (t = −0.07) — zero — where
it read +0.0011 (t = +2.72) at v7.2x. The clip toward too-low ceilings had been
landing on the mislabelled rows it was clipping onto.

That is not a reason to drop Grabit, and the evaluation protocol changed to say
so. Grabit censors ~5% of rows, so a pooled test divides its effect by twenty and
mistakes dilution for weakness. The suite now prints a **zone scorecard** over the
rows Grabit exists for — those paid ≥90% of their own ceiling:

| Grabit zone (n=65) | Baseline | Grabit |
|---|---|---|
| MAE | $7.27M | **$6.52M** |
| bias | −$6.85M | −$6.43M |
| rows better / worse | — | **59 / 6** |

The keep/drop rule for Grabit now reads this zone alone: keep it while the zone
MAE delta is negative, drop it when the zone itself turns positive. The pooled
`t > 2` rule still governs challenger changes — features, filters,
hyperparameters — which act on every row.

**A guard to watch.** The locked confirmation split has drifted apart from the
selection pool over these two versions: 0.7611 / 0.7461 (gap +0.0150) at v7.2x
against 0.7643 / 0.7415 (gap +0.0228) at v7.4x. The levels are not comparable to
each other — different players, different difficulty — but the *trend* is the
signal the split exists to give, and n=186 puts a single reading well inside
noise. Neither change looks like metric-mining (v7.3x was a semantic fix; v7.4x
*lowered* pooled R²), so no action now. At the next version bump, re-score each
accepted change on confirmation rows only; if accepted changes are systematically
≤0 there while >0 on the selection pool, tighten the protocol so selection
metrics exclude the confirmation split.

### v7.5x: the early-supermax list

v7.4x widened the supermax award window to `s-3` to reach two extensions signed
two summers before they began. The window was too blunt: it handed 35% ceilings
to eleven rows, and two of them were genuine 30% max signings — Klay Thompson
2019 and De'Aaron Fox 2026 fell to 86% of an inflated ceiling and dropped out of
the censoring zone entirely. `ISSUES.md` had warned against exactly this
("do not widen the award window to paper over it").

Early-signed supermaxes run about one per year and are high-profile, so they are
enumerated instead:
`data/raw/raw_external/early_supermax.csv` carries John Wall 2019 and
Karl-Anthony Towns 2024 with the qualifying award in a note column. All four
sentinel cases now sit at their true ceilings and inside the zone; over-cap rows
stay at 0. Headline movement is nil (A1 0.7609 → 0.7610) because only four rows
change hands — the point is that the right four rows are censored.

**The systematic fix is a signing date per contract**, which Spotrac exposes and
the scrape does not yet take. Until then the list is the honest instrument.

**Committing that CSV exposed a dead whitelist.** `.gitignore` excluded
`data/raw/` — the directory itself, so git never descended into it and no `!`
rule below could fire. The two hand-curated reference tables the model depends
on, `awards_full.csv` and `player_draft_2020-2025.matched.corrected.csv`, had
therefore **never been in version control**: CLAUDE.md's warning that "a clone
that lacks these does not reproduce the published numbers" described the state
of every clone. The exclusion is now `data/raw/*` and both tables are tracked
(`0f667cd`). Without them `_compute_max_eligible` silently degrades to an age
heuristic.

### v7.6x: year-1 rows above their tier ceiling

No fresh signing can exceed its tier maximum. A year-1 label on a salary above
that ceiling is therefore *proof* the row is a later year of an older deal, not
a judgement call — so `_filter_mislabeled_year1` drops it at load in every
trainer and in the suite.

The test deliberately uses the **tier-only** ceiling, exposed as
`tier_ceiling_pct` alongside the floored `max_eligible_pct`. The no-decrease
floor added in v7.4x exists to legalise escalator pay for Stage-2 scoring, and
escalator pay is exactly what a fresh contract cannot be; including it would
make the test vacuous.

Six rows fall: Curry, Paul, Westbrook, Wiggins and McCollum 2019, Wall 2020.
Rule-based rather than a data edit, so it survives rebuilds and refreshes.

| Layer | v7.5x | v7.6x |
|---|---|---|
| A1, common rows | 0.7589 | **0.7635** (+0.0046) |
| A1, headline | 0.7610 | 0.7635 |
| A2 2024-26 | 0.8368 | 0.8269 |
| B1 forward | 0.8239 | 0.8200 |
| Grabit zone | n=67, MAE $6.59M | n=61, MAE $6.45M (59/2) |

The row set changes (1297 → 1291), so **the common-row line is the honest one**;
the A2/B1 dips are fold-reassignment noise, since six rows confined to 2019-2020
cannot move a 386-row 2024-26 slice by a point.

**These six were the visible tip of a much larger class.** A probe against
`salaries_prehistory.csv` found that of the 88 year-1 2019 rows with 2018 pay on
record, **38 step by an escalator-shaped ratio** — LeBron at exactly 1.050, Paul
George and Joel Embiid at exactly 1.080 — and 23 of those sit above $15M. The
aggregate said the same thing: 2019 carried 255 year-1 rows against 166-206 in
every later season.

### Protocol change: decisions read the selection pool only

*No version number — the evaluation protocol changed, no published figure moved.*

The v7.4x entry promised to re-score accepted changes on confirmation rows at
the next bump. Done, and the signal held. Over v7.2x → v7.5x, on identical rows
with identical fold assignment:

| Slice | v7.2x | v7.5x | Δ |
|---|---|---|---|
| Selection pool (n=1107) | 0.7568 | 0.7603 | **+0.0035** |
| Confirmation split (n=184) | 0.7533 | 0.7481 | **−0.0052** |

Difference-in-differences on row-level squared-error improvement, bootstrapped
by player cluster: **+5.6 × 10⁻⁵, 95% CI [+8.5, +111.1] × 10⁻⁶ — excluding
zero**, and it survives removing the six v7.6x rows. Dozens of accept/reject
reads had begun fitting the rows they were watching.

`oof_groupkfold` now returns a second fold × seed matrix computed on
selection-pool validation rows only; `paired_delta` for accept/reject reads that
matrix, and both persist in `evaluation_suite.json` (`fold_r2_selection` is
decision-grade, `fold_r2` is context). Confirmation rows still serve as training
data in other folds — they are excluded only from the metric that decides.
Headline A1/A2/B1 stay pooled for continuity.

The honest limits: the CI's lower bound sits near zero, and one plausible
channel is innocent — v7.3x's fill table is estimated from the whole dataset,
85% of which is selection-pool players. The evidence is *moderate and
directionally clear*, not a verdict. Tightening costs 7% of the decision sample,
so it is a free precaution either way.

### v7.7x: continuations, and why one signal was not enough

The class v7.6x exposed spans every season, not just 2019 — the salary-chain
detector breaks mid-deal whenever ratios drift. The instrument is the anchored
contract span from `scripts/refresh_spotrac.py`: a row covered by a contract
that *starts earlier* is a continuation, whatever its label says.

**The first cut used the span alone and was wrong.** It deleted 346 rows, and
7.2% of them appeared in that season's actual Spotrac FA-signings list —
Jalen Brunson 2022, Fred VanVleet 2023, Jimmy Butler 2019, all unambiguously
fresh signings. Spotrac's Free-Agent anchor is unreliable for contracts later
superseded by an extension, so span evidence alone deletes real market prices.

`_filter_continuations` demotes a row only when **three independent signals
agree**:

1. **span** — the salary-matched covering contract starts in an earlier season
   (AAV within 25% of the row's pay, so a renegotiated deal matches its new
   money rather than the superseded shell);
2. **step** — pay moved from last season by an escalator-shaped ratio
   (0.92-1.081; raises cap at 8% of year-1 salary), with last season's pay read
   from the full table plus `salaries_prehistory.csv` at the 2019 boundary.
   Brunson stepped 15×, VanVleet 1.9× — both clear instantly;
3. **veto** — the row is absent from that season's FA-signings list.

Rows without last-season pay on record are kept. Precision over recall: a stale
price in training is cheaper than a deleted real one.

119 rows fall (2019: 35, tapering to 2026: 1). The largest are all verifiable
mid-contract seasons — LeBron 2019, Ben Simmons 2021, Gordon Hayward 2023.

| Layer | v7.6x | v7.7x |
|---|---|---|
| A1, common rows | 0.7545 | **0.7647 (+0.0102)** |
| A2 2024-26 | 0.8269 | 0.8431 |
| B1 forward | 0.8200 | 0.8283 |
| Grabit zone | n=61, MAE $6.45M | n=57, MAE $6.14M (53/3) |

**+0.0102 is the largest single paired gain in the v7.x line** — three times the
rookie-scale fill, nine times the two-stage pipeline itself. It is also a
correctness change, not a modelling one: what improved is that the model stopped
being asked to explain prices that were never set in the season they were filed
under.

The confirmation canary moved *with* this change (confirmation 0.786 against
selection 0.761), which is what a genuine cleanup looks like and not what
metric-fitting looks like.

### v7.8x: the CBA floor is a bound too

Grabit had censored one side for eight versions. The other bound was visible in
plain sight: the champion overpredicted the 297 rows pinned at the veteran
minimum by **+$2.38M, with 93% of rows overshot**. The league floor props those
players' pay above their unconstrained price, so the observation is a *ceiling*
of the latent — the exact mirror of a max contract, where the observation is a
*floor* of it.

**This is not the idea the oracle experiment killed.** That one treated *good*
players on minimums as right-censored, and it fails economically: a good player
on a minimum could have signed elsewhere for more, so his salary is a choice,
not a constraint. The population here is the opposite one — players whose
unconstrained price sits *below* the minimum, held up by a rule no contract can
cross. The oracle result says nothing about it.

Mechanics, all mirrors of the right side:

- `_compute_floor` derives `is_at_floor` (Minimum-labelled rows inside the
  vet-min band, ≤2.5% of cap) and `floor_pct`, a (season, experience-bucket)
  lookup recovered from the data's own mass points — 0.0148 for two-year vets,
  0.0235 for ten-year vets — rather than maintained CBA tables.
- The left gate mirrors the albatross gate: censor only rows the baseline also
  prices near the floor (`pred ≤ k × observed`, screened over k ∈ {1.5, 2.0,
  3.0}, adopted at 2.0). A ring-chasing veteran priced well above his pay stays
  uncensored, so his latent keeps meaning.
- Stage 2 becomes `clip(latent, floor_pct, max_eligible_pct)`.
- `floor_gate_k=0` reproduces v7.7x exactly.

Results, 10 seeds, against the one-sided champion on identical rows:

| | v7.7x | v7.8x | 95% cluster CI |
|---|---|---|---|
| **Floor zone** MAE (n=297) | $2.40M | **$1.98M** | per-row \|err\| **[−0.469, −0.374]** |
| Floor zone bias | +$2.37M | +$1.93M | 254/43 better/worse |
| **Pooled MAE** (n=1172) | $3.181M | **$3.065M** | **[−0.139, −0.091]** |
| Pooled bias | +$0.118M | −$0.116M | |
| Spillover, non-floor rows | — | −$0.012M | [−0.035, +0.012] — includes 0 |
| A1 / A2 / B1 | 0.7647 / 0.8431 / 0.8283 | 0.7653 / 0.8465 / 0.8324 | selection t = +1.39 |

**The two headline verdicts differ, and the arithmetic explains why.** R² weights
by squared error, so 297 rows whose dollar errors are small contribute almost
nothing to it; MAE weights by row, and those rows are 25% of the sample. The
same intervention is large under one metric and invisible under the other. Under
the project's stated rule — R² as the headline, MAE as a guard — this passes on
the guard and on its zone, exactly as the right side does.

**Spillover is favourable here, unlike the right side.** The rows neighbouring
the floor were also overpredicted, so the shared-tree pull points the right way;
the max side taxes its 50-75% neighbours instead. Fixed-row segments: worst
non-floor |bias| growth +$0.07M against the $0.30M gate.

**A scare that dissolved.** A predicted-band decomposition first read
"−$1.21M on 2-4% others", which looked like serious damage. It was
band-composition shift — the same regression-to-the-mean artifact that layer C
exists to avoid, reappearing on the model-comparison axis instead of the
calibration axis. Fixed-row segments showed every non-floor band flat or better.
**When comparing two models, define the segments on fixed rows, never on either
model's predictions.**

### v7.9x: continuation filter v2 — dated spans

The three-signal filter (v7.7x) caught the easy cases — rows whose escalator
step, span, and FA-list veto all agreed — but left the mid-contract years where
the step signal failed. Two data-layer fixes widened the span instrument's
reach, and the filter was then swapped from a three-signal consensus to a
dated-span test.

**The 10-day parse bug.** `parse_contracts` read a 10-day deal's terms string
`10 yr(s) / $85,578` as a 10-YEAR contract, fabricating phantom spans back to
1980. `spotrac_signing_types.csv` 8,954 → 5,769 rows; 328 phantom >5yr
contracts removed (326 at 10yr, 2 legitimate 6yr deals from the 2005-CBA era).
The continuation filter still dropped the same 119 rows afterward — no phantom
span was load-bearing. Evidence:
`docs/briefs/2026-07-24-continuation-filter-v2.RESULT.md`.

**Span fixes.** Three corrections to `contract_spans`:

- **Option-aware anchor**: Spotrac's `fa_year` on an option-final deal is the
  option-decision summer, so the span was starting one season early. The fix
  slides the span onto the option season when the option year exceeds the
  nominal end (49 spans affected, 14 change start — Embiid, Doncic, Gobert,
  Butler and 10 more).
- **Extension fallback +1**: unmatched extensions start paying at
  `signing_season + 1`, not `signing_season` (24 spans affected).
- **Renegotiation carve-out**: a renegotiation-and-extend re-prices its signing
  season to market, so that season is **fresh** even though an older span covers
  it (6 pairs — Turner 2022, Sabonis 2023, Clarkson 2023, Isaac 2024,
  Markkanen 2024, JJJ 2025).

With correct spans, `_filter_continuations` now demotes a row outright when a
dated contract covers it and starts before the season, unless the season is
renegotiated-fresh or the row appears in that season's FA-signings list; rows the
dates cannot decide fall back to the unchanged three-signal consensus. Frame
**1,172 → 949** (342 demoted — 335 by dated span, 7 by fallback — 0
contradicted by any instrument).

| Layer | v7.8x (1,172 rows) | v7.9x (949 rows) |
|---|---|---|
| A1 | 0.7653 | 0.7631 |
| A2 | 0.8465 | 0.8160 |
| B1 | 0.8324 | 0.8191 |

**These headlines are not comparable** — different row sets, different R²
denominators (D1). The change itself was judged on the 949 common rows:

| instrument | result |
|---|---|
| Common-row paired A1 (selection) | −0.0009, t −0.13 — **neutral** |
| C2 max \|bias\| growth | +$0.16M — **pass** |
| `is2019` contamination control | +0.0035 (t 2.29) → −0.0004 (t −0.91) — **eliminated** |

**A pure-removal change cannot be judged on common-row A1**, which scores only
rows both frames keep and is by construction blind to the benefit of removing
stale rows. The pre-registered success criterion was the `is2019` season-offset
signal that ISSUES #8 measured as the signature of the remaining stale prices —
it collapses to noise on the new frame. The common-row neutrality confirms the
change did no harm; the `is2019` collapse is where the benefit shows up.

Tags: `v7.9x` `ddf4bae`.

### v7.10x: `prev_cap_pct` repair + 2026 cap correction

The model's second-strongest anchor was fabricated for a large minority of rows.
`phase3.build_contract_features` derived `prev_cap_pct` by looking back inside
the 2019+ training table only, so any fresh signing whose previous contract began
before 2019 — all of season 2019, plus later vets coming off long pre-2019
deals — took the rookie-scale slot fill instead of its real prior pay (Klay
Thompson 2019: 0.037 for a true 0.186). The fix has two parts.

**Part 1**: scrape the 30 BBRef 2018-19 team salary tables (extend-only, +399
players, 0 conflicts). Season-2019 prior coverage goes from **28% → 96%**; the
6 still missing are two-way/minimum players absent from the team salary table,
plus the Enes Kanter → Enes Freedom rename.

**Part 2**: after the year-1 lookback loop, fill every remaining row from
observed previous-season pay — the same union of the salary table and
`salaries_prehistory.csv` that `_load_prev_season_cap_pct` already anchors the
ceiling rule on — leaving the slot fill only for true first contracts. Klay
2019 → 0.186, escalator rows → their real priors, 2024 stays 0.300.

| instrument | ref (buggy) | Arm A (repair) | paired |
|---|---|---|---|
| A1 selection | 0.7631 | 0.7732 | **+0.0135 ± 0.0018, t +7.60** |
| A2 | 0.8160 | 0.8232 | same direction |
| B1 forward | 0.8191 | 0.8255 | same direction |
| C2 max \|bias\| growth | — | +$0.16M | pass |
| 2019 segment bias | −$0.74M | −$0.42M | halved |

**ΔSel above +0.01 (the red-flag line) — explained, not spurious:** this repairs
the #2 feature on the largest season's rows (2019 prior coverage 28%→96%) plus
every escalator. Corroborated by the B1 forward lift and free of leakage
(prior-season pay is strictly ex ante). Evidence:
`docs/briefs/2026-07-24-prev-cap-pct-fix.RESULT.md`.

**2026 cap correction.** The configured cap sat at a stale projection; the
official value is **$164,961,000**. Five extension-max salaries that were stale
in lockstep failed to trigger `check_caps.py` — an instance of the
"stale-in-lockstep" failure mode, where wrong cap ÷ wrong salary still produces
the expected tier share and the cross-check passes. The check now reconciles
7/11 on-tier at the official cap.

**Frame 949 → 944.** Extending `salaries_prehistory.csv` made five 2019
continuations newly decidable and demoted (Millsap, Snell, Galloway, Okafor,
Ferrell — all genuine escalator continuations, none in the FA list). This is
ISSUES #2's documented decidability gain.

Rookie-exit anchor arms (flag / NaN / slot-average) all failed a gate and were
not adopted. The breakout-class bias (−$3.16M) is assigned to the max
classifier.

Tags: `v7.10x` `(see git tag)`.

### v7.11x: `prev_cap_pct` = previous-season pay (feature batch)

Five candidate feature arms scored against the 14-feature Grabit champion on the
944-row frame. Only one moved the model: **Arm C**, which replaces
`prev_cap_pct`'s "previous contract's year-1 pay" semantics with a uniform
"previous-season actual pay" — a one-function swap in
`build_contract_features(prev_mode="prev-season")`. Evidence:
`docs/briefs/2026-07-25-feature-batch.RESULT.md`.

| arm | what | ΔSel | t (fold) | verdict |
|---|---|---|---|---|
| **c** | prev_cap_pct = prev-season pay, uniform | **+0.00178** | **+1.96** | **adopted (boundary)** |
| a | +9 trend columns (native NaN) | −0.00015 | −0.05 | rejected — flat |
| b_replace | production features ← as-of-signing | −0.03218 | −4.40 | rejected — harmful |
| d1 | +est_value_prev | +0.00008 | +0.03 | rejected — flat |
| e_new | +rookie_award_tier ordinal | +0.00017 | +0.15 | rejected — flat |

**Adoption rationale for Arm C at the boundary** (per-fold t = 1.96, a hair
under the t > 2 gate): (i) the magnitude +0.00178 reproduces the pre-registered
+0.0020 from the prev-cap RESULT; (ii) A1, A2, B1 and MAE all improve in the
same direction; (iii) C2 is pristine (max fixed-segment |bias| growth +$0.031M);
(iv) it is a semantics simplification rather than added capacity — the model now
reads one thing ("what were you paid last season?") instead of conditionally
reading year-1-of-the-prior-deal or last-season, depending on where the lookback
landed.

**Protocol correction.** The prev-cap RESULT's "t = 7.0" for the same comparison
was **seed-paired** — seed sd ≈ 0.001 inflates t roughly 3× relative to the
canonical fold pairing. The magnitude reproduces; the t does not. **The pairing
unit is always the fold**, per the worker brief; seed-to-seed variation measures
RNG, not signal.

**Rejected arms and why.** Trend features, prior-year estimated value, and
rookie-award ordinal are all flat because DARKO already encodes trajectory and
the existing awards channel already carries the pedigree signal.
Stats-as-of-signing (Arm B replace) is actively harmful: the market prices
expected growth, so current-season production beats signing-date production for
extensions — breakout bias moved −$2.87M → −$5.06M, and only MPJ 2022
(priced pre-injury) is the rare inversion.

Tags: `v7.11x` `(see git tag)`.

### v7.12x: designated-ceiling award-path fix (ISSUES #19)

`_compute_max_eligible` granted the 35% Designated-Veteran / 30% Rose ceiling
from All-NBA + experience alone, blind to the CBA's team-continuity requirement.
12 genuine maxes wore a ceiling one tier too high:

- **35% → 30%** (changed teams, acquired on vet deals, or non-designated):
  Kawhi 2019, AD 2020, Kemba 2019, Kyrie 2019, Mitchell 2025, Beal 2021,
  Sabonis 2023/24, Randle 2022, Brunson 2025.
- **30% → 25%** (rookie extension paid the 25% base; the Rose escalator is a
  later year, and KAT's award window was off-by-one): KAT 2019, Tatum 2021,
  Ja 2023, Holmgren 2026, J. Williams 2026.

New curated `data/raw/raw_external/designated_ineligible.csv` — the inverse of
`early_supermax.csv` — gates the award path in `_compute_max_eligible`. No team
signal exists in the frame, so a curated list is the safe fix pending a
signing-date team-match test. Frame unchanged at 944.

| metric | before | after |
|---|---|---|
| Max zone (`is_max`) | 56 | **68** |
| Grabit zone MAE | $6.18M | **$4.44M** |
| A1 (same 944 rows) | 0.7849 | **0.7878** |
| Confirmation | — | 0.8090 |

Evidence: the landing commit `61bdfcd`.

Tags: `v7.12x` `61bdfcd` · `v7.13x` `981e6dc`.

### v7.13x: real service years + the correctness debt

`_compute_max_eligible` derived service years as `season − draft_year`, falling
back to `age − 19` for the 24% of rows without a draft entry. The fallback
systematically over-tiered undrafted and late-entry players: Austin Reaves
(undrafted, debuted 2021) read as 8 years of service when he had 5, landing on
a 30% ceiling instead of 25%; his 2026 salary sits at exactly 25.000% of the
cap, making him a max contract the model was missing. The fix scrapes per-player
debut seasons from BBRef's player index pages — the same source as `height.py` —
covering 99.5% of training-data players.

119 base-tier ceilings moved (117 down, 2 up), max zone 68 → 70. A second
provably mis-tiered row became the max it is: Butler 2019 at exactly 30.000% of
the cap, added to `designated_ineligible.csv` for the same team-change reason as
Kawhi, Kyrie, and Kemba in v7.12x. Paired ΔSel +0.00001 (t 0.08) — a
correctness fix, as expected, since the ceiling is not a model feature and
affects only the censoring mask and the Stage-2 clip.

Landed alongside four other correctness items, all adopted on the same
principle — a wrong fact is fixed regardless of its metric sign, provided it
does not actively hurt (the pre-registered threshold was t > −2):

- **Awards name-join repair.** `norm()` in `build_external_features` did not
  strip footnote marks (`^`, `st1`, `covid2`, etc.) from `awards_full.csv`
  names. 83 of 943 award rows were dirty, covering every ROY winner from 2015
  onward plus MVP/All-NBA seasons for Durant, Curry, Giannis, Jokic, and others.
  148 training rows recovered award mass. Paired ΔSel −0.00076 (t −0.72) — the
  recovered mass is redundant with DARKO and the affected players are
  ceiling-pinned, so the correction adds variance without lift.
- **Kanter alias.** `enes kanter` in `salaries_prehistory.csv` vs `enes freedom`
  in the training data; the join missed, so Kanter's 2019 `prev_cap_pct` was the
  rookie-scale fill (7.3%) instead of his true prior pay (18.9%). Added
  `player_name_aliases.csv` (also catches `wesley iwundu` → `wes iwundu`).
- **Canonical 10-seed CV figure.** `train.py` now writes the 10-seed average R²
  into `grabit_results.json`, replacing the single-seed value. No model change.
- **`predict.py` rewired to the champion stack.** Was using plain XGBoost with
  no censoring and no Stage-2 clip; now runs the same Grabit + CBA-bound pipeline
  as `export_web.py`.

| Layer | v7.12x | v7.13x |
|---|---|---|
| A1 | 0.7878 | 0.7865 |
| A2 | 0.8335 | 0.8341 |
| B1 | 0.8303 | **0.8342** |
| Grabit zone MAE | $4.44M (n=68) | **$4.30M (n=70)** |
| Confirmation | 0.8090 | 0.8070 |

B1 0.8342 is the highest forward R² the project has recorded.

Evidence: `docs/briefs/2026-07-26-service-years.RESULT.md`,
`docs/briefs/2026-07-26-cleanup-debt.RESULT.md`.

## Phase 8: Three-stage pipeline (signing route + new features)

| Ver | Model | A1 (told) | A2 (told) | B1 (told) | N | Feat | Change |
|-----|-------|-----------|-----------|-----------|---|------|--------|
| 8.0x | XGBoost (Grabit v4) | 0.7989 | 0.8515 | 0.8356 | 944 | 14 | Stage 3 — the signing route enters the model. Push + extension clip in one composition module. Ex-ante A1 0.7865 (clip only) |
| 8.1x | XGBoost (Grabit v4) | 0.8127 | 0.8588 | 0.8419 | 944 | 14 | Impact-source join repaired on `nba_id`. 2026 forward origin 0.7992 → 0.8311 |
| **8.2x** | **XGBoost (Grabit v4)** | **0.8210** | **0.8625** | **0.8528** | **944** | **15** | **`is_waived`, the fifteenth feature + missingness-semantics repair** |

### Convention change at Phase 8

**From v8.0x onward, headline A1/A2/B1 are told-route numbers.** The model is
scored knowing the signing route, which the champion also had and merely
ignored. The ex-ante figure that continues the v7.1x-v7.13x series is the
clip-only arm's A1 0.7865 at v8.0x. Do not compare the two naively.

Tags: `v8.0x` `a28b221` · `v8.1x` `29b6383` · `v8.2x` `800bed7`.

### v8.0x: Stage 3 — the signing route enters the model

Two layers, composed in one module (`src/model/stages.py`), all three consumers
pointed at it: `latent -> push -> clip(lo, hi) -> stage 3`. The push moves
rows the model prices below their ceiling toward it where P(max) >= 0.52;
Stage 3 returns extension rows to their legal raise cap. The refactor alone is
inert -- the clip-only arm reproduces the previous champion on every metric.

**Landed with it**: the signing-season award anchor in `_compute_max_eligible`
(the same wrong fact that was found in its second home in the extension-cap
module) and the rebuilt `prev_cap_pct` feature, together worth +0.0002
(t = 0.79) -- correctness, as expected.

**Costs recorded, not argued away.** Paired dSel +0.00856 at t = 1.11, adopted
on the legal-bound grounds v7.4x/v7.9x/v7.13x set. C1 relative calibration
fails by 0.0024 because the push only ever moves rows upward. And on the
pooled metric the push alone is negative (-0.0057) with all of the gain coming
from Stage 3 (+0.0143, t = 2.28). The push is kept because it is judged where
it acts -- max zone $4.30M -> $3.18M -- and because its row-level audit nets
-$8.33M of absolute error over the 8 rows it touches.

Confirmation split read at this bump: 0.8309 against a selection pool of 0.7928
-- the canary improved more than the pool.

| Layer | v7.13x (ex ante) | v8.0x (told route) |
|---|---|---|
| A1 | 0.7865 | **0.7989** |
| A2 | 0.8341 | 0.8515 |
| B1 | 0.8342 | 0.8356 |
| 2024 / 2025 / 2026 | 0.8757 / 0.8173 / 0.7964 | 0.8781 / 0.8167 / 0.7992 |

### v8.1x: impact-source join repaired on nba_id

Same 944 rows, same 14 features, told-route convention unchanged from v8.0x.
Pure correctness: three impact sources (DARKO, LEBRON, LAKER/RAPM) spelled the
same player differently and the name-season merge left those rows split, so
league-median minutes, usage and availability reached the model in place of the
truth. Joining on `(nba_id, season)` and filling age from the player's own
invariant history removes that.

**The 2026 forward origin moves 0.7992 -> 0.8311**, the largest single-origin
gain in the project's record, because the repaired rows are concentrated in the
recent seasons the holdout scores.

Confirmation split read at this bump: 0.8339 against a selection pool of 0.8085.

| Layer | v8.0x | v8.1x |
|---|---|---|
| A1 | 0.7989 | **0.8127** |
| A2 | 0.8515 | 0.8588 |
| B1 | 0.8356 | 0.8419 |
| 2024 / 2025 / 2026 | 0.8781 / 0.8167 / 0.7992 | 0.8679 / 0.8225 / 0.8311 |

### v8.2x: `is_waived`, the fifteenth feature + missingness repair

Same 944 rows, told-route convention unchanged. Over v8.1x's 14-feature
0.8127 / 0.8588 / 0.8419, the feature adds a paired dSel of +0.00666 at
t = 2.29 -- the first new feature in a long run to clear t > 2 with every
guardrail clean, and it does it on a fact the model previously had no way to
see: a minimum signed within 365 days of a buyout is supplemental pay on top of
a guarantee the old team is still carrying.

**The coverage control is why this one counts.** A bare "has a usable Spotrac
page" indicator is worth +0.00029 (t = 1.02), so the gain is not the collection
channel that killed the supply features at v7.x. `is_waived_known` is kept as
an audit column and deliberately not shipped despite scoring slightly better.

**v8.2x's +0.0083 over v8.1x is not all `is_waived`.** A
missingness-semantics repair landed inside the same tag, and its own isolated
harness (holding `is_waived` in both arms so the two do not mix) measures
+0.0088 selection-paired at t = 1.10. Both belong in this entry, separately
attributed.

**What the missingness repair fixes is semantics, not more data.** A player who
did not play has an **undefined** usage rate, not a missing one -- 46 full-table
rows now carry `games = minutes = 0` and a `did_not_play` status instead of a
league-median fill. Availability uses the exact s/s-1/s-2 window, skips unknown
seasons with weight renormalisation instead of scoring them as zero, and keeps
the 2020-21 season at 72 games. A Basketball Reference advanced-stat fallback
fills absent workload and box rates over eight seasons, matched on name-season
then on the stable `player_url`, filling only gaps and never synthesising RAPM.
After it, age, height, mpg and availability have zero missing values in the 944
frame.

Accepted on correctness. The worker's own RESULT records the exception rather
than burying it: A2 R2 falls 0.0018 (while A2 MAE and bias improve) and
relative calibration exceeds its guard by 0.0017. The evidence that it works is
where it claimed: the 100 previously-missing rows fall from $3.94M to $3.00M
MAE and 2023 R2 goes 0.789 -> 0.843.

Confirmation split read at this bump: 0.8521 against a selection pool of 0.8152
-- the canary is again ahead of the pool.

| Layer | v8.1x | v8.2x |
|---|---|---|
| A1 | 0.8127 | **0.8210** |
| A2 | 0.8588 | 0.8625 |
| B1 | 0.8419 | 0.8528 |
| 2024 / 2025 / 2026 | 0.8679 / 0.8225 / 0.8311 | 0.8711 / 0.8359 / 0.8499 |

### v8.3x: rookie-contract filter (exp <= 1)

Frame shrinks 944 → 868. Players in their first two NBA seasons (exp 0 or 1)
are removed from training: their contracts are slotted by draft position, not
negotiated by the market, so they carry no market-pricing signal. The filter
uses `exp = season - debut_season` where `debut_season` comes from the signing
date's `signing_season` field (1,100 rows) with fallback to the paying season
(197 rows). One unknown-debut row (Jeff Dowtin 2023) is kept.

A1 0.8091 (from 0.8210) — the raw drop is a denominator change, not a
regression. Common-row paired delta is −0.0003, indistinguishable from zero
under the fold reshuffle (ISSUES #35). Accepted on correctness: rookie-scale
contracts are not market observations.

| Layer | v8.2x (944) | v8.3x (868) |
|---|---|---|
| A1 | 0.8210 | **0.8091** |
| A2 | 0.8625 | 0.8606 |
| B1 | 0.8528 | 0.8395 |

### v8.4x: `mpg_x_waived` interaction (16th feature)

Gate override: the waiver interaction measured t = 1.96 on the 944-row frame,
just below the t > 2 threshold. Override justified because:
(1) placebo arms (permuted `is_waived`) score zero — effect is real;
(2) only 11 relevant rows in the top prior-pay band, 4 locked in the
confirmation split — oracle ceiling is t = 1.09, the frame physically cannot
reach t > 2 on this segment;
(3) mechanism is well-understood: a waiver erases ~81% of price history
(OLS slope 0.750 → 0.140).

Both arms tested on the 868-row frame. Arm A (`prev_cap_pct × is_waived`)
regressed A1 by −0.0007. Arm B (`mpg × is_waived`) won on every metric.
Adopted Arm B.

| Layer | v8.3x | v8.4x |
|---|---|---|
| A1 | 0.8091 | **0.8099** (+0.0008) |
| A2 | 0.8606 | 0.8617 (+0.0011) |
| B1 | 0.8395 | 0.8403 (+0.0008) |
| 2026 origin | 0.8417 | 0.8452 (+0.0035) |

### v8.5x: Wall contract-structure repair + error-board universe fix

Data repair, no model change. John Wall's 4-year supermax (signed 2017,
effective 2019) was split by the mid-contract trade to Houston into two fake
year-1 rows in `contract_structure_v2.csv`. Hand-verified fix: 2019 → yr 2/4,
2020 → yr 3/4, 2021 → yr 4/4. Net −1 training row on the v8.4x frame (Wall
2019; Wall 2020 was already caught by the tier-ceiling mislabel filter).

The error board investigation that surfaced Wall also surfaced its own bug:
`diagnostics.py` built its OOF universe with a partial filter chain (year-1 +
rookie-scale only), so rows the model never trains on appeared as model errors
— 7 of the board's 40 entries were such phantoms (Hayward 2019/2023, Graham
2020, Millsap 2019, Love 2022, Camara 2025, Wall 2020), all left-censored
pre-2019 contracts or rookie deals the chain drops. Both partial chains in
`diagnostics.py` now apply the full train.py membership chain, and the new
`scripts/make_error_board.py` builds the board from `oof_reference.csv`
(champion OOF, told route) — the previous board scored Stage-1 raw predictions,
overstating errors the deployed pipeline does not make (Zubac 2025 $19.3M →
$0.0M under the Stage-3 raise cap; Beal 2022 and Jaylen Brown 2024 vanish under
the Stage-2 max push). Review state persists in `error_board_kicked.csv`.

| Layer | v8.4x (n=868) | v8.5x (n=867) |
|---|---|---|
| A1 | 0.8099 | 0.8147 |
| A2 | 0.8617 | 0.8627 |
| B1 | 0.8403 | 0.8446 |
| 2026 origin | 0.8452 | 0.8429 |

Row set changed by the repair itself, so the columns are not a paired
comparison — recorded for continuity, not as a gain claim.

### v8.6x: `playoff_mpg_diff` (17th feature), on a decoupled route classifier

Two changes ship together because the first was blocking the second.

**The feature.** `playoff_mpg_diff` = playoff minutes per game minus
regular-season `mpg`, **0.0** where the player did not appear in that season's
playoffs. A playoff rotation is a dated, public judgement made by the party with
the most information, on the games that matter most, and made before the
contract this row prices — the market reads it and the regular-season box score
does not contain it. 480 of 867 rows played; the work lands where the mechanism
says it should. Benched zone (`diff < -5`), champion stack, **selection rows
only** (n=139, per ISSUES #20a): MAE **$2.470M → $2.237M**, 70 rows better
against 38 worse; pooled (n=158, reporting) $2.511M → $2.275M. The deep bench
(`diff < -10`, n=68) moves $2.280M → $1.838M. The 387 untouched no-playoff rows
move $3.158M → $3.156M, so nothing leaks onto the rest of the frame. Minutes are
scraped from BBRef's playoff per-game pages by `scripts/scrape_playoff_mpg.py`
into `data/processed/playoff_mpg.csv` and attached at load time by
`src/features/playoff_minutes.py`, the same light-touch pattern as the waiver
interactions — no training-data rebuild, so no unrelated column moves under the
measurement.

**One column, not two: the `po_games` adjudication.** A two-column variant
adding `po_games` scored *better* (+0.00594, t 2.61 against +0.00485, t 2.12 on
the plain-XGB frame; C−A paired +0.00109, t 4.89) and was rejected anyway, as a
team-success proxy — the family METHODOLOGY already rejects at +0.0004 (`win_pct`,
`made_playoffs`). Three falsifications agree: permuting `po_games` **within
(season, playoff team)**, which preserves each team's run length exactly and
destroys only the player's own deviation, retains the entire edge (+0.00117
permuted against +0.00109 real); the player-specific half alone
(`po_games − team_max_po_games`) scores −0.00045 (t −1.58); and `po_games` adds
+0.00007 (t +0.36) over a pure team-level column. corr(`po_games`, team playoff
depth) is +0.78, and 57% of matched rows played every game of their team's run.
`playoff_mpg_diff` survives the same test — the within-team permutation retains
6% of its gain (+0.00028, t +0.72), a team-mean control retains 10% — which is
why one column ships and the other does not. **Do not re-open this.**

**The blocker (ISSUES #44).** `route_mixture.attach_clf_features` built the
Stage-2 route classifier's input list as `list(FEATURE_COLS) + CLF_EXTRA_COLS`,
so a 17th regression feature silently became a 37th *classifier* feature. That
coupling is not neutral: P(max) feeds a knife-edge gate (push only where
P ≥ TAU = 0.52), and a column with no route content still perturbs the
classifier's trees through `colsample_bytree=0.8`. Measured with the column
force-fed (5 folds × 10 seeds), the classifier does not get *worse* — max-class
AUC 0.9828 → 0.9830, multiclass log-loss 0.7604 → 0.7539, mean |ΔP| 0.003 — which
is exactly why this was invisible. The damage is **26 (row, seed) gate decisions
flipping on 10 borderline rows** (max |ΔP| 0.154), two flipping even on the seed
average, and they land where a push is worth millions: Chet Holmgren 2026 gains
one, Jimmy Butler 2023 loses one, Jamal Murray 2025 — a true max paid *at* his
ceiling — loses his in one seed of ten, Sabonis 2020 ($19.80M against a $27.29M
ceiling) gains one in two. Net −0.00215 paired (t −1.25), concentrated in one
fold: a coin flip on expensive rows, not a signal. It cost the feature its gate —
t 1.51 with the column in the classifier against t 2.71 without, at the suite's
seeds.

The fix states the classifier's list explicitly: `CLF_BASE_COLS`, today's
sixteen, frozen, plus the curated `CLF_EXTRA_COLS`. Adding a regression feature
now leaves OOF P(max) **bit-identical** (verified per (row, seed): max |ΔP| =
0.000e+00, zero gate flips) and the suite still prints 36 classifier features.
A column joins the classifier only by a deliberate edit to `CLF_BASE_COLS`.

**Gates** (867-row frame, told-route champion, GroupKFold(5) by player, seeds
42-51, both arms in one process on identical folds; the classifier decoupled so
the two arms differ only in the regression list):

| gate | limit | measured | |
|---|---|---|---|
| 1 paired selection t | > 2 | **+0.00576, t +3.41** (5/5 folds positive) | PASS |
| 2 A2 same direction | > 0 | +0.00442 | PASS |
| 3 C1 relative calibration | ≤ +0.005 | −0.00360 (`\|slope−1\|` shrinks) | PASS |
| 4 C2 worst mechanism segment | ≤ +$0.30M | +0.091 pooled / +0.105 selection (Sign & Trade, n=13) | PASS |
| 5 B1 (large drop vetoes) | — | 0.8418 → 0.8431 | PASS |

| Layer | v8.5x | v8.6x |
|---|---|---|
| A1 | 0.8147 | **0.8209** (+0.0062) |
| A2 | 0.8630 | 0.8674 (+0.0044) |
| B1 | 0.8416 | 0.8427 (+0.0011) |
| 2024 / 2025 / 2026 origin | 0.8634 / 0.8176 / 0.8430 | 0.8647 / 0.8193 / 0.8432 |
| MAE | $2.93M | $2.87M |
| D3 locked confirmation | 0.8070 (selection 0.8157) | 0.8200 (selection 0.8208) |

The locked confirmation split gains more than the selection pool (+0.0130
against +0.0051), so the canary is ahead of the rows the decision watched — the
opposite of the pattern the 2026-07-23 adoption audit caught.

Both columns are suite runs at the suite's own seeds 0-9; the v8.5x column is
the incumbent re-measured on this tree (`evaluation_suite_prev.json`), not the
figures published in the v8.5x entry, which read A2 0.8627 / B1 0.8446 from an
earlier run of identical code. Two runs of the same code recording B1 0.8446 and
0.8416 is the scale of run-to-run noise at n=305 forward rows — which is exactly
why the decision above is a **paired, same-process** comparison rather than a
difference of two published numbers.

### v8.7x: Spotrac phantom-data cleanup + waiver recovery rebuild

**Data correction, not a model change — frame definition moved.**

Two fixes landed in one commit pair (8071745, 0c2acaf):

1. **ISSUES #41 re-scrape**: the 16 junk Spotrac redirect pages (generic
   homepage content cached as player pages) had injected 1064 phantom
   signing_types rows for 173 unrelated players. The continuation filter used
   this phantom data both ways — incorrectly dropping 64 genuine Year-1
   signings (LeBron 2019, Paul George 2019, Robert Covington 2019-2023) and
   incorrectly keeping 35 actual continuations (P.J. Tucker, Allen Crabbe,
   Lou Williams). 15/16 pages re-fetched with correct URLs; Josh Gray has no
   Spotrac page. Spotrac CSVs rebuilt from all 510 cached pages.
2. **ISSUES #40 waiver recovery**: `_resolve_waiver_no_signing()` baked into
   training_data_v2.csv via `rebuild_training_data.py`. 65 rows resolved via
   conservative date windows. Waiver coverage on the evaluation frame:
   827/896 known (was 756/867).
3. **ISSUES #36/#38/#31 data repairs** (committed in 8071745): salary
   corrections, contract-structure overrides, min_cap_charge — baked into the
   rebuild.

Frame: 867 → **896** (+29 net: +64 gained, −35 lost). R² not comparable
across frame sizes (#35).

| Layer | v8.6x (n=867) | v8.7x (n=896) |
|---|---|---|
| A1 | 0.8209 | 0.8021 |
| A2 | 0.8674 | 0.8569 |
| B1 | 0.8427 | 0.8479 |
| 2024 / 2025 / 2026 origin | 0.8647 / 0.8193 / 0.8432 | 0.8561 / 0.8430 / 0.8412 |
| MAE | $2.87M | $3.02M |
| D3 locked confirmation | 0.8200 (sel 0.8208) | 0.7895 (sel 0.8036) |

A1 and A2 drop mechanically — the 29 new rows include several low-salary
players (Bronny James, Adem Bona, Dalen Terry at ~1.4% cap) that add variance
to the denominator. B1 improves because the forward-test frame (2024-2026)
gained from the cleanup: n=305 → 318 (+13 rows), and the model now trains on
correctly classified training data. MAE growth ($2.87→$3.02M) reflects the
larger frame mixing in harder-to-price rows.

### Post-v8.7x: data repairs and evaluation tooling (no version consumed)

Four ISSUES resolved, training data rebuilt. No model or feature change — no
version number consumed.

| Commit | What | ISSUES |
|--------|------|--------|
| `966f176` | RAPM re-scrape: 58 rows filled via `nba_id` matching (name mismatches). Coverage: 2023 81.8%, 2026 83.5%. Remaining gaps confirmed absent from nbarapm.com | #39 amended |
| `966f176` | min_cap_charge batch 2: 6 corrections (MCW 2019, Gerald Green 2019, Barea 2019, Iwundu 2020, Bates-Diop 2023, Hyland 2026). `fix_minimum_convention.py` dedup logic fixed: append-only, never re-detects existing corrections | #38 amended |
| `5fa5f4a` | Luol Deng 2019: 4th stretched dead money case ($5M LAL stretch, played for MIN). `salary_override` to vet minimum $2,564,753 | #36 updated (3→4 rows) |
| `5fa5f4a` | `abs_bias_growth()` and `zone_scorecard()` promoted to `evaluate_suite.py` as shared helpers. `diagnostics.py` signing-type residuals split by confirmation/selection. `eval_floor_branch.py` imports shared versions | #20 partial |
| `1ec3a48` | Nunn continuation: verified already fixed in commit `8071745`. Similar-case scan: OG Anunoby 2021 defensible, PJ Dozier 2020 negligible | #31 closed |

Training data rebuilt on the corrected 896-row frame. Metrics unchanged from
v8.7x: A1 = 0.8019, A2 = 0.8573, B1 = 0.8493, MAE = $3.02M.

Signing-type bias on the 896-row frame (OOF, selection-only):

| Type | N | Bias | MAE |
|------|---|------|-----|
| Bird Rights | 248 | −$2.2M | $4.5M |
| Sign & Trade | 10 | −$5.1M | $5.7M |
| Cap Space | 59 | −$0.7M | $3.4M |
| Early Bird | 44 | −$1.4M | $2.5M |
| MLE | 110 | +$1.0M | $2.7M |
| Non-Bird | 11 | +$1.9M | $2.2M |
| Minimum | 206 | +$2.5M | $2.5M |

The systematic pattern — Bird Rights underpredicted, MLE/Minimum overpredicted
— is the motivation for Stage 3 signing-type work (in progress).

(That table is a plain-XGBoost OOF, not the champion stack; the champion's own
biases are smaller. See ISSUES #47 and the v8.8x entry below.)

### v8.8x: the Stage-3 signing-type offset

**A told-parameter correction, not a feature.** The stack carried a systematic
Signing Residual by mechanism that no feature removes. Rung 1 of the correction
adds **one constant per signing type** to the composed prediction, then puts the
row back inside the law:

```
pred_corrected = clip( minimum( pred + offset(type), ext_cap ), floor, ceiling )
offset(type)   = n / (n + k) x mean OOF residual of that type,   k = 20
```

Neither the Grabit regression nor the route classifier is refit — the correction
is pure post-processing of the champion's own predictions, so champion and
candidate share every fold, every seed and every fitted model, and the paired
delta carries zero fit noise.

**Only the four ELIGIBILITY mechanisms are corrected**: Bird Rights, Cap Space,
Early Bird, Non-Bird. MLE, BAE, Minimum, Sign & Trade, Other and Unknown take a
zero offset and come out bit-identical (asserted, max |diff| = 0.000e+00 over
479 rows). That exclusion is a **leakage ruling made before any score on the arm
was seen**, not a scoring choice: an exception mechanism is determined by the
contract value itself — a deal is "the MLE" because of what it pays — so
conditioning on it reads the target. It costs the two largest biases on the
board: Minimum's +$1.97M and Sign & Trade's −$3.29M stay exactly where they are.

`k = 20` is pre-registered in `stages.SIGNING_K` beside TAU and MARGIN. A k = 0
arm was computed REFERENCE-ONLY and decided nothing.

**Fold honesty.** Layer A corrects fold *f* from the OOF residuals of rows
outside fold *f*, per (fold, seed) cell, before the seed average. Layer B learns
season *T*'s offsets from an inner GroupKFold OOF over seasons < *T* only, so B1
never sees the season it scores in any capacity. Offsets are estimated over all
rows including confirmation (an offset is a fitted parameter and confirmation
rows already sit in every training fold); the deciding metric excludes them
(ISSUES #20a).

| Layer | v8.7x (n=896) | v8.8x (n=896) |
|---|---|---|
| A1 | 0.8015 | **0.8126** (+0.0111) |
| A2 | 0.8574 | 0.8672 (+0.0098) |
| B1 | 0.8495 | 0.8578 (+0.0083) |
| 2024 / 2025 / 2026 origin | 0.8567 / 0.8470 / 0.8410 | 0.8692 / 0.8456 / 0.8568 |
| MAE | $3.03M | $2.94M |
| C1 slope | 0.952 | 0.942 |
| D3 locked confirmation | 0.7897 (sel 0.8029) | 0.7885 (sel 0.8161) |

Paired on the selection pool, by fold, same fitted models: the signing offset
alone is **+0.01287, se 0.00245, t +5.25**. Where the intervention acts — MAE
over the four eligible types, selection pool, paired by fold — it is
**+$0.201M, t +2.48** ($3.673M → $3.472M), with |bias| reduction +$0.771M
(t +3.07). Per-type bias, pooled:

| Type | n | bias v8.7x | bias v8.8x | MAE v8.7x | MAE v8.8x |
|---|---|---|---|---|---|
| Bird Rights | 282 | −$1.74M | −$0.36M | $3.83M | $3.60M |
| Cap Space | 71 | −$0.49M | −$0.20M | $3.70M | $3.69M |
| Early Bird | 50 | −$1.25M | −$0.41M | $2.57M | $2.43M |
| Non-Bird | 14 | +$1.96M | +$1.42M | $2.93M | $2.83M |

368 of the 417 eligible rows move (mean |move| $1.25M); legality clips the
offset entirely away on the other 49. Zero rows below the floor, above the tier
ceiling, or above a binding raise cap, in either layer.

**Two caveats, both recorded rather than argued away.**

1. **The confirmation canary (n = 55) does not corroborate the win.** MAE is
   flat — **+$0.01M, t = +0.07** — while signed bias moves **−$0.07M → +$1.03M**.
   The correction over-corrects rows where the champion was already unbiased. On
   the suite's pooled-OOF reading of the same 55 rows it is slightly negative
   (−$0.05M, bias +$0.18M → +$1.34M). Fifty-five rows decide nothing either way,
   but the direction is the one that would show up first if the offsets were
   fitting the frame rather than the market.
2. **B1's forward all-rows signed bias moves +$0.33M → +$1.03M** while R² and
   MAE both improve (0.8495 → 0.8578, $3.30M → $3.25M). The offsets are
   estimated on seasons < T and applied to season T, so any drift in the market's
   per-mechanism premium shows up as over-correction on the newest season —
   which is exactly what origin 2025 does (R² 0.8470 → 0.8456, the one origin
   that goes backwards). Watch this number at the next version bump: a
   correction whose bias keeps growing while its MAE stops improving has stopped
   correcting and started shifting.

**Provenance of the deployed constants.** `stages.SIGNING_OFFSETS_DEPLOYED`
(Bird Rights +0.012605, Cap Space +0.002419, Early Bird +0.007097, Non-Bird
−0.006075, all cap_pct) is the k=20, all-OOF-rows form measured on this frame by
`scripts/eval_stage3_signing.py` on 2026-08-06. They are **data-dependent
constants** — a training-data rebuild invalidates them the way it invalidates a
published R². Regeneration is now step 5 of CLAUDE.md's refresh block; ISSUES
#48 tracks making it mechanical. The evaluation suite is not exposed (it
estimates its own offsets every run); the deployed single-fit consumers are.

`predict.py` is **bit-identical to v8.7x** and correctly so: an unsigned free
agent has no signing mechanism, so every offset on that path is 0.0. The
deployed offsets are passed in explicitly and the identity is asserted, so the
no-op is demonstrated by the code. `scripts/export_web.py` has not been wired
(ISSUES #46) — the site still shows the v8.7x composition.

Both A1/A2/B1 above are **told-route numbers** under the 2026-07-26/27
convention, now told-MECHANISM as well: told "he re-signed on Bird Rights", the
price is still the whole problem, so the offset passes the convention's test the
same way the extension clip does. The suite prints all four arms — clip only
(v7.13x), + push, + extension clip (v8.7x), + signing offset (champion).

Evidence: `scripts/eval_stage3_signing.py`,
`outputs/models/stage3_signing_offset_eval.json`,
`outputs/models/evaluation_suite.json`. The suite reproduces the gating
measurement to **0.000e+00** on A1, A2 and B1.

### Corrections to earlier findings

- **The residual-by-salary-tier table reported in v7.0x was a statistical
  artifact.** Binning residuals by the actual target produces a monotone bias
  gradient even for a perfectly calibrated model. Binned by predicted value
  instead, bias is flat within ±$0.65M. The model was never systematically
  underpaying stars by $6M.
- **Extending Tobit censoring to the lower bound is a dead end for accuracy** —
  *for the population that finding tested.* Feeding back a fold-honest
  `P(mechanism | x)` scores −0.0073; a leaky oracle with the realised mechanism
  gains only +0.0137. The Minimum/MLE residual gap is the spread of a bimodal
  `y | x`, not recoverable error.

  **v7.8x sharpened the scope of this.** What is dead is treating *good* players
  on minimums as right-censored: they could have earned more elsewhere, so their
  salary is a choice and the censoring premise is false. Players whose
  unconstrained price sits *below* the minimum are the opposite case — the floor
  is a rule no contract can cross — and left-censoring them is worth $0.42M per
  row on 297 rows. Read the original finding as being about *mechanism as a
  proxy for unobserved choice*, not about bounds in general.

### Routes tested and rejected

Five route-mixture architectures were measured across three phases and all five
are closed. The numbers are recorded here so nobody re-runs them.

**Route mixture, ex ante (full six-route form).** Fails all five gates: dSel
-0.0479, t = -2.50, calibration slope 0.819. The fault localises to `V_max`
and `V_floor`, which sit far from `f(x)` on every row, so composing them with
an imperfect P redistributes across the ~93% of rows that are not candidates.
Evidence: `docs/briefs/2026-07-26-extension-route.RESULT.md`.

**Floor branch.** Oracle headroom is +0.0396 and unreachable: P(floor) is
anti-ranked against the error (Spearman -0.611 inside the zone); the
classifier's most confident quarter of the zone carries 2.5% of the
over-prediction where an error-ordered selector would carry 78.7%. The twelve
worst floor rows score **above the average non-floor row** on minutes, all
three impact metrics, usage, prior pay and awards -- the same features that
make the model overprice them make them invisible. Evidence:
`docs/briefs/2026-07-27-floor-branch.RESULT.md`.

**Offseason-injury signal.** 1 of 9 fat-tail floor rows has a dated offseason
injury. The rest are age/decline, playstyle devaluation, buyout dynamics and
cold markets. A perfect injury flag is worth +0.0097, a quarter of the floor
headroom. No-go on the scrape.

**MLE branch.** P(mle) AUC 0.71: whether a player is offered an exception
depends on the signing team's cap position, which no player feature sees.
Evidence: `docs/briefs/2026-07-25-route-mixture.RESULT.md`.

**Ex-ante per-route delta.** Fails on surface lift; mean P(bird) is 0.29 on
every row, so P x delta_bird moves the whole price surface. Evidence:
`docs/briefs/2026-07-26-route-delta.RESULT.md`.

## Key milestones

```
Phase 1    Phase 2 (leaked)     Phase 3         Phase 4        Phase 5    Phase 6    Phase 7        Phase 8
Ridge      Ridge/XGB            Cleanup         Tuning         Features   Grabit     Data + protocol  3-stage + route
                                                                                     
0.43 ─→ 0.62 ─→ 0.68 ─→ ···    0.65 ─→ 0.73    0.75 ─→ 0.75   0.76       0.76       0.758 → 0.787    0.799 → 0.813
 v1.0     v2.2    v3.1  ···      v4.0    v4.2x    v5.0x   v5.3x   v6.2x    v7.0x      v7.1x    v7.13x   v8.0x    v8.8x
                        ↗ 0.87                                                                                 (current)
                  Leaked features                                                                told-route numbers
                  (removed in v4.0)                                                              from v8.0x onward
```

Phase 7 looks flat through v7.8x and then jumps. The v7.1x-v7.8x plateau was by
design: six of those eight entries are correctness changes -- refreshed data,
corrected caps, contaminated rows removed, a semantic fill, honest ceilings, four
rows re-censored -- and two of them *lowered* the headline. v7.9x-v7.13x continue
that pattern (four filter/repair changes, one label fix, and a cleanup batch)
but the compounding finally shows: v7.13x's B1 0.8342 was the highest forward R2
the project had recorded, and every point came from removing something wrong
rather than adding modelling complexity.

Phase 8 introduces a convention change (told-route numbers from v8.0x), the
three-stage pipeline, and the first new feature since v6.2x. The correctness
thread continues: six of the last nine versions landed on a correctness argument
rather than a metric win, and several did not clear t > 2: v7.9x (common-row A1
-0.0009), v7.13x (t = 0.08), v8.0x (t = 1.11), v8.1x, and the missingness
repair inside v8.2x (t = 1.10). The standing rule these encode: **a wrong fact
is repaired on correctness; a suboptimal parameter must clear the gate.**
v8.8x is the other kind: four fitted parameters, so it had to clear the gate
where it acts, and did (+$0.201M eligible-type MAE, t +2.48; A1 paired t +5.25).

The count of known-wrong things is the better progress line: an optimistically
biased test set, two 9%-wrong season targets, 259 prorated rows, a fill 4.6x
too high for undrafted players, 13 impossible ceilings, 342 escalator years
filed as fresh signings, a fabricated prior-pay feature on 28% of a season's
rows, 12 mis-tiered max contracts, 119 over-tiered service-year ceilings, 83
dirty award-join rows, a name alias dropping a $20M prior, a signing-season
award anchor wrong in its second home, split impact-source rows filling league
medians in place of real data, 46 did-not-play rows carrying fabricated
usage rates, 1064 phantom signing-type rows from 16 junk Spotrac redirect
pages, 4 stretched dead-money salaries filed as signed contracts, and a
minimum-contract convention mixing paid and cap-charge values across seasons
-- all closed or under active correction.

v7.10x's `prev_cap_pct` repair (+0.0135, t 7.6) is the largest single paired
gain in the project's recent history -- repairing the #2 feature on the largest
season's rows. v7.12x's max-zone MAE drop (6.18 -> 4.44) is the largest zone
improvement, from relabelling 12 genuine maxes the award path had mis-tiered.
v8.0x continues that zone improvement (4.30 -> 3.18 with the push + Stage 3)
and v8.2x's B1 0.8528 sets the forward-R2 high-water mark.

## Notes

**R² comparability**: three boundaries in this table are not crossable.
Phase 1–2 (v1.0–v3.5x) versus Phase 3+ (v4.0+) differ in training filters and
leaked features — the apparent v3.5x 0.866 → v4.0 0.645 drop is that, not a
regression. Phase 7 versus everything before it differs in both the row set and
the target values for two seasons. R²'s denominator moves with the dataset, so
**whenever a training filter changes, the comparison has to be run on a fixed
evaluation set** rather than by reading two R² figures off this table.

**Phase 2 leaked features**: `is_vet_min`, `is_mle_range`, `is_rookie_scale` are derived from the target variable (salary determines contract type). They gave large R² gains in training but are unknowable at prediction time. All removed in v4.0.

**v6.2x award design**: Step decay [1, 0.85, 0.65, 0.4, 0.1] outperforms exp decay (0.85/yr) and no-decay across all data sources. Filtering to 2017+ removes noisy older data. All-Star selections add noise (−0.0014 CV); team awards (All-NBA, All-Defensive, All-Rookie) and individual awards are the useful signal.
