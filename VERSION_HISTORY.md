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
| **7.8x** | **XGBoost (Grabit v4)** | **0.7653** | **0.8324** | **1172** | **14** | **Two-sided Grabit — left-censoring at the CBA floor. Pooled MAE $3.18M → $3.07M** |

### Versioning convention

Every change that moves a published number — data, target, features, model, or
evaluation protocol — takes the next `vN.Mx` and is tagged on its landing commit
with the headline A1/A2/B1 in the tag message, so any quoted figure can be
reproduced with a checkout. Bug fixes count (v5.3x was one); the *kind* of change
belongs in the Change column, not the numbering. Diagnostics, tooling and
documentation changes do not consume a number.

Tags: `v7.1x` `09dbe4e` · `v7.2x` `cb27319` · `v7.3x` `b31a6fb` · `v7.4x` `ecdf3da` ·
`v7.5x` `c17880d` · `v7.6x` `d7c72ef` · `v7.7x` `63e290c` · `v7.8x` `c03eb9a`.

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

## Key milestones

```
Phase 1    Phase 2 (leaked)     Phase 3         Phase 4        Phase 5    Phase 6    Phase 7
Ridge      Ridge/XGB            Cleanup         Tuning         Features   Grabit     Data + protocol
                                                                                     
0.43 ─→ 0.62 ─→ 0.68 ─→ ···    0.65 ─→ 0.73    0.75 ─→ 0.75   0.76       0.76       0.758 → 0.765
 v1.0     v2.2    v3.1  ···      v4.0    v4.2x    v5.0x   v5.3x   v6.2x    v7.0x      v7.1x    v7.8x
                        ↗ 0.87                                                                (current)
                  Leaked features
                  (removed in v4.0)
```

Phase 7 looks flat and mostly is, by design: six of its eight entries are
correctness changes — refreshed data, corrected caps, contaminated rows removed,
a semantic fill, honest ceilings, four rows re-censored — and two of them
*lowered* the headline. The count of known-wrong things is the better progress
line: an optimistically biased test set, two 9%-wrong season targets, 259
prorated rows, a fill 4.6× too high for undrafted players, 13 impossible
ceilings, and 125 escalator years filed as fresh signings, all closed.

Two entries did move the model rather than the data. v7.7x is the largest paired
gain of the phase (+0.0102) precisely *because* it is a correctness change — the
model stopped being asked to explain prices that were never set in the season
they were filed under. v7.8x is the only genuine modelling change since v7.0x,
and its effect is invisible in R² and plain in MAE ($3.18M → $3.07M, CI
excluding zero): it fixes 297 small-dollar rows that squared error barely
notices.

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
