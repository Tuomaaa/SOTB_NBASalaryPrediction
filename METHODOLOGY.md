# Methodology — NBA Free Agent Valuation Model

## Objective

Predict a player's market value as **cap_pct** (annual salary / salary cap), using prior-season performance data. Inspired by Hollinger's BORD$, but fully open-source and reproducible.

## Data Sources

| Source | Data | Rows |
|--------|------|------|
| Basketball Reference | Per-season salary, age, team | 4,723 player-seasons (1,086 players, 2019–2031) |
| nbarapm.com | DARKO DPM, LEBRON, LAKER, usage, box-score rates | 3,880 player-seasons (903 players) |
| Basketball Reference | Height in inches | 5,416 players |
| Basketball Reference | Pre-window salaries for the ceiling floor and `prev_cap_pct` | 767 player-seasons (~550 players, 2016–2018) |
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
7. **Mislabel filter** (v3.0.5): drop year-1 rows paid above their *tier* ceiling — provable escalator mislabels. Reduces to **1,291 rows**. See below.
8. **Continuation filter** (v3.0.6): drop rows a three-signal consensus identifies as later years of older deals. Reduces to **1,172 rows**. See below.
9. **GroupKFold CV**: 5-fold, same player stays in same fold to prevent within-player leakage.

The last two filters exist because the contract-structure table tags a
pre-2019 contract's *first observed* season as year 1 — the detector never saw
the seasons before the data starts — and the salary-chain detector also breaks
mid-deal whenever escalator ratios drift. Both classes put escalator prices into
training wearing fresh-signing labels.

### Sample composition

The 2026 season carried 55 usable rows before the 2026-07-22 refresh and 124
after. The pre-refresh set was not merely smaller, it was **optimistically
biased**: it contained only extensions and players already under contract, all
of which are easier to price than an open-market signing. Adding the actual free
agent signings dropped the 2026 forward R² from 0.843 to 0.811 on a test set
that finally represents the task.

### Prorated salaries (removed in v3.0.1)

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

### Mislabelled year-1 rows (removed in v3.0.5)

No fresh signing can exceed its tier maximum, so a year-1 label on a salary above
that ceiling is *proof* the row is a later year of an older deal — not a
judgement call. `_filter_mislabeled_year1` drops six: Curry, Paul, Westbrook,
Wiggins and McCollum 2019, Wall 2020.

The test uses the **tier-only** ceiling, exposed as `tier_ceiling_pct` beside the
floored `max_eligible_pct`. The no-decrease floor exists to legalise escalator
pay for Stage-2 scoring, and escalator pay is exactly what a fresh contract
cannot be, so including it would make the test vacuous.

### Continuation rows (removed in v3.0.6)

The six above were the visible tip. Of the 88 year-1 2019 rows with 2018 pay on
record, 38 stepped by an escalator-shaped ratio — LeBron at exactly 1.050, George
and Embiid at exactly 1.080 — and 2019 carried 255 year-1 rows against 166-206 in
every later season.

The instrument is the anchored contract span from `scripts/refresh_spotrac.py`.
**A span alone is not sufficient evidence** — a span-only filter deleted 346 rows
of which 7.2% appeared in that season's actual FA-signings list (Brunson 2022,
VanVleet 2023, Jimmy Butler 2019), because Spotrac's Free-Agent anchor is
unreliable for contracts later superseded by an extension. `_filter_continuations`
demotes a row only when three independent signals agree:

1. **span** — the salary-matched covering contract starts earlier (AAV within 25%
   of the row's pay, so a renegotiated deal matches its new money rather than the
   superseded shell);
2. **step** — pay moved from last season by an escalator-shaped ratio
   (0.92-1.081), reading `salaries_prehistory.csv` at the 2019 boundary;
3. **veto** — the row is absent from that season's FA-signings list.

Rows without last-season pay on record are kept: precision over recall, since a
stale price in training is cheaper than a deleted real one. 119 rows fall; the
largest are verifiable mid-contract seasons (LeBron 2019, Simmons 2021,
Hayward 2023). Common-row A1 rose +0.0102, the largest paired gain of Phase 7.

**v3.1.1 replaced the primary path with a dated-span test.** With correct spans
(see below), `_filter_continuations` now demotes a row outright when a dated
contract covers it and starts before the season, unless the season is
renegotiated-fresh or the row appears in that season's FA-signings list; the
three-signal consensus stays the fallback for rows no dated contract covers.
Frame 1,172 → 949 (342 demoted — 335 by span, 7 by fallback — 0 contradicted).

#### Span rules

Spotrac's `fa_year` on an option-final deal is the **option-decision summer**,
not the start season, so the correct span is `[fa_year − years + 1, fa_year]`;
the naive formula started one season early. An unmatched extension starts paying
at `signing_season + 1`. Of the 1,953 dated contracts, 120 block anchors (6.1%)
predate their own signing and are date-resolved — a block anchor alone is never
a trustworthy span.

#### Renegotiation convention (adjudicated 2026-07-24)

A renegotiation-and-extend re-prices its signing season to market, so that
season is **fresh** even though an older span covers it. Six pairs in the data:
Turner 2022, Sabonis 2023, Clarkson 2023, Isaac 2024, Markkanen 2024, JJJ 2025.
JJJ 2025 was the one acceptance-list mismatch: the brief expected it to stay
continuation, but its 2025-26 salary was raised $11.6M by the renegotiation on
2025-07-13 — structurally identical to Markkanen 2024 and Turner 2022. Resolved
as fresh.

## Feature Set (21 features)

### Performance Metrics (z-scored within season)
| Feature | Description |
|---------|-------------|
| `darko_dpm_z` | DARKO Daily Plus-Minus, z-scored by season |
| `lebron_z` | LEBRON metric, z-scored by season |
| `laker_z` | LAKER, z-scored by season |
| `kalman_filtered_stats` | Kalman-filtered player quality estimate. See below. |
| `darko_od_diff_z` | DARKO offensive minus defensive component, z-scored by season. See below. |
| `lebron_od_diff_z` | LEBRON offensive minus defensive component, z-scored by season. See below. |
| `laker_od_diff_z` | LAKER offensive minus defensive component, z-scored by season. See below. |

#### O/D diff z-scores (v5.1.0)

The three composite z-scores (DARKO DPM, LEBRON, LAKER) measure overall player
quality. The O/D diffs capture a second, **orthogonal** dimension: whether a
player tilts toward offense or defense. `darko_od_diff = darko_odpm - darko_ddpm`
(and likewise for LEBRON and LAKER), then z-scored within season so the scale
matches the composites.

**Why the diffs, not the raw O/D columns.** Replacing the 3 composites with
6 raw O/D z-scores (Arm A) scored +0.0044, t = 1.14 on 20 seeds — the gains
are unstable because 6 columns cost 3 degrees of freedom while mostly
re-expressing the same total quality signal. The diffs (Arm B) are additive —
keep all existing features, add 3 — and the signal they carry (offensive tilt)
is nearly uncorrelated with the composites (confirmed: kf_offense ↔ kf_defense
r = −0.186). Paired ΔSel = +0.0026, t = 2.01 on 20 seeds; 4 of 5 folds
positive; stable across seed halves (seeds 0-9: t = 1.73, seeds 10-19: t = 2.08).

A two-filter Kalman variant (one filter from O metrics, one from D) was also
tested (Arm D): it recovers only half of `kalman_filtered_stats`'s value
(r = 0.775 with the composite) because independent filters lose the
cross-information between O and D measurements.

Code: `src/features/base_rating.py::attach_od_diffs()`, attached at load time
in `train.py::load_training_data()`.

### Age & Workload
| Feature | Description |
|---------|-------------|
| `age` | Player age at time of season |
| `age_squared` | Captures nonlinear decline after peak |
| `mpg` | Minutes per game (minutes / games) |
| `usage_pct` | Usage rate (offensive load) |
| `availability_3yr` | Weighted GP% over past 3 seasons (0.5/0.3/0.2) |
| `playoff_mpg_diff` | Playoff minutes per game minus regular-season `mpg`, **0.0** where the player did not appear in that season's playoffs. See below. |

#### `playoff_mpg_diff` (v4.3.0)

A playoff rotation is a public, dated judgement of a player, made by the party
with the most information, on the games that matter most, and made *before* the
contract this row prices. The market reads it; the regular-season box score does
not contain it. On the 867-row frame 480 rows played (55.4%) and 158 sit in the
benched zone (`diff < -5`), where the champion's MAE falls **$2.470M → $2.237M**
on selection rows. The 387 non-playoff rows, which carry the 0.0 fill, move
$3.158M → $3.156M — the zone rule does the work and nothing leaks onto the rest
of the frame.

Zero, not NaN, is the right fill: "his team missed the playoffs" is a fact about
the team, not a gap in the player's record, and a zero difference is the neutral
reading. Minutes are read from BBRef's playoff per-game tables via
`scripts/scrape_playoff_mpg.py` and attached at load time by
`src/features/playoff_minutes.py`.

**Only the difference ships.** A two-column variant adding `po_games` scored
better (+0.00594, t 2.61 against +0.00485, t 2.12 on the plain-XGB frame) and
was rejected as a **team-success proxy**, the family already rejected at +0.0004
(`win_pct`, `made_playoffs`): permuting `po_games` within (season, playoff team)
— preserving each team's run length exactly and destroying only the player's own
deviation — retains its entire edge (+0.00117 permuted against +0.00109 real),
the player-specific half alone scores −0.00045 (t −1.58), and `po_games` adds
+0.00007 (t +0.36) over a pure team-level column. `playoff_mpg_diff` survives
the same falsification: the within-team permutation retains 6% of its gain
(+0.00028, t +0.72) and a team-mean control retains 10%.

#### `kalman_filtered_stats` — Kalman-filtered player quality (v5.0.0)

A Kalman filter over the three impact metrics (DARKO DPM, LEBRON, LAKER) that
produces a single filtered quality estimate per player-season. The filter runs
causally per player — it sees only seasons up to and including the current one —
so it is strictly ex ante and carries no look-ahead.

**State model.** A player's latent quality evolves as
`x_{t+1} = x_t + drift(age) + process_noise`, where `drift(age)` is a piecewise
constant estimated from mean year-over-year change bucketed by age: +0.21
(age <= 22), +0.05 (22-25), -0.04 (25-28), -0.19 (28-31), -0.26 (31-34),
-0.27 (34+). Each season's three z-scored metrics are noisy measurements of the
same latent state: `[darko_z, lebron_z, laker_z] = x + measurement_noise`.

**Parameter estimation, from metric data only.** All filter parameters are
estimated from the metric data, never from the target:

- **R** (measurement noise covariance) from pairwise metric disagreement:
  DARKO 0.138, LEBRON 0.203, LAKER 0.447. LAKER is downweighted roughly 3x
  relative to DARKO, which matches its known noisiness (smaller sample of
  possessions, heavier regularisation).
- **drift(age)** from the mean year-over-year change in each metric, bucketed
  by age band and averaged across the three metrics.
- **Q = 0.1592** (process noise), the residual year-over-year variance after
  drift removal, minus the measurement noise contribution.

**Why it outperforms raw metrics.** `corr(kalman_filtered_stats, darko_z) = 0.959` — high, but
the residual carries two things the raw z-scores do not: (1) **optimal
multi-metric weighting**, where LAKER is downweighted relative to DARKO and
LEBRON in proportion to its measurement noise, and (2) **age-aware drift
prediction**, so a 22-year-old's filtered state is pulled upward by the
expected growth trajectory while a 32-year-old's is pulled down. The
falsification confirms this: a simple exponential moving average (EMA) over the
same three metrics, which does the smoothing without the weighting or the drift,
recovers only half the gain (+0.00263, t = 1.17 against kalman_filtered_stats's +0.00450,
t = 2.78).

Code: `src/features/kalman_quality.py`, attached at load time in
`train.py::load_training_data()`.

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
| `kf_market_value` | Kalman-filtered market trajectory (v5.2.0, replaces `prev_cap_pct`). For a player at season T: anchor a random-walk KF at the market's last observed price, update through model-predicted intermediate seasons. Three-tier anchor: (1) most recent Year-1 eval-frame row, (2) earliest rookie-scale season for first-rounders, (3) fallback to `prev_cap_pct`. At inference time, a base model (21 features with `prev_cap_pct`) prices intermediate seasons — no circularity because the measurement model never sees `kf_market_value`. See `src/features/kf_market_value.py`. |
| `is_waived` | 1 when Spotrac records a waiver or buyout in the fixed 365 days before the signing that prices this row. Events after signing are excluded. Unknown source/signing coverage remains auditable through `is_waived_known`, which is not a model feature. |
| `mpg_x_waived` | `mpg × is_waived` (v4.2.0). A waiver erases most of a player's price history — the OLS slope of pay on prior pay drops 0.750 → 0.140 across it — so the market re-prices him off current workload instead. NaN where `is_waived` is unknown, never 0. |

#### Filling `prev_cap_pct` for first contracts (v3.0.2)

A first contract has no previous deal to observe. Until v3.0.2 the fill was one
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

### Missingness semantics (v4.1.0)

A player who did not play has an **undefined** usage rate, not a missing one.
46 full-table rows now carry `games = minutes = 0` and a `did_not_play` status
instead of a league-median fill. Availability uses the exact s/s-1/s-2 window
and **skips unknown seasons with weight renormalisation** rather than scoring
them as zero; 2020-21 counts 72 games.

A Basketball Reference advanced-stat fallback fills absent workload and box
rates over eight seasons, matched on name-season then on the stable
`player_url`, filling only gaps and never synthesising LAKER. After the repair,
age, height, mpg and availability have zero missing values in the 944 frame.

The impact-source join itself was repaired in v4.0.1: the three sources
(DARKO, LEBRON, LAKER) sometimes spelled the same player differently, and
the old name-season merge left those rows split. Joining on `(nba_id, season)`
closes the gap.

### Features Tested and Rejected

| Feature | ΔCV R² | Reason Rejected |
|---------|--------|-----------------|
| `bpm`, `ts_plus`, `fg3_plus`, `threepar_plus` | < +0.001 | Redundant with DARKO/LEBRON/LAKER |
| `ast_tov_ratio` | +0.0001 | Redundant with ast_pct |
| `experience_years` | -0.0004 | Collinear with age |
| `win_pct`, `made_playoffs` | +0.0004 | Team context already in player metrics |
| `playoff_bpm_diff_adj`, `playoff_laker_diff_adj` | +0.0003 | Sparse (35% coverage), noisy. **Superseded 2026-08-01, but only for minutes**: `playoff_mpg_diff` (adopted v4.3.0, +0.00576 paired, t 3.41) shows the playoff signal is in the ROTATION, not in playoff box-score rates. A benched player's per-possession metrics stay respectable on a small sample; the minutes themselves are what the market prices. These two rows stand as measured — do not re-test them expecting the v4.3.0 result. |
| `po_games` (alongside `playoff_mpg_diff`) | +0.00109 paired, t 4.89 | Rejected 2026-08-01 as a **team-success proxy** despite clearing the bar: permuting it within (season, playoff team) retains the entire edge. Same family as `win_pct` / `made_playoffs` below. See the `playoff_mpg_diff` section above. |
| `max_eligible_pct` | +0.0008 | Collinear with age/experience |
| `is_vet_min`, `is_mle_range` | N/A | Derived from target variable (leakage) |
| `is_rookie_scale` | N/A | Handled by rookie scale filter instead |
| `team_value_B` | +0.0001 | Franchise value doesn't predict individual salary |
| `injury_reports` | +0.0002 | Redundant with availability_3yr |
| Spotrac injury burden | -0.0017 to -0.0024 directional paired Delta selection R-squared | Rejected 2026-08-31 after a three-seed, five-fold screen. One-year and two-year time decay, log games missed, fixed injury-type severity, recurrence weighting, raw games missed, and event count were all negative. The full type-plus-recurrence score was -0.00239 at t = -3.20. The 10-seed gate was stopped by user decision after every candidate arm and every recurrence fold was negative. Spotrac dates the missed-game interval rather than the diagnosis, so Damian Lillard's April 2025 Achilles tear first appears on 2025-10-22 and is unavailable for his 2025-07-19 signing. See `docs/briefs/2026-08-31-injury-feature.RESULT.md`. |
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
| `height_x_age` | +0.0004 (t = 0.47) | Noise, and it *hurts* the segment it targets: only 36 tall+old rows in training, split between minimum ring-chasers and productive bigs on real contracts. The interaction can only push one direction ("old+tall = cheaper"); the impact metrics already separate declining bigs from productive ones per player |
| `weight_x_age` | N/A | No weight column in the training set — the height scraper reads only the height field of the BBRef index pages. Would inherit the same counter-effect problem as `height_x_age`: heavy+old contains both ends of the price range |
| `kf_innov` (Kalman innovation) | +0.00167 (t = 1.34) | Tested alongside `kalman_filtered_stats` (v5.0.0). The filter's innovation (surprise) — actual measurement minus predicted measurement — captures how much a player over- or under-performed expectations. Does not clear t > 2 on its own; the quality estimate already absorbs the signal |
| `ema_quality` (exponential moving average) | +0.00263 (t = 1.17) | Control for `kalman_filtered_stats` (v5.0.0). A simple EMA over the three z-scored impact metrics, which does temporal smoothing without optimal metric weighting or age-aware drift. Recovers only half of `kalman_filtered_stats`'s gain, confirming the Kalman filter's advantage is in the weighting and drift, not mere smoothing |
| 6 raw O/D z-scores replacing 3 composites | +0.0044 (t = 1.14) | Arm A of the O/D split ablation (v5.1.0). Replacing `darko_dpm_z`, `lebron_z`, `laker_z` with their 6 offensive and defensive components costs 3 degrees of freedom and mostly re-expresses total quality. Gains are fold-unstable: range −0.00000 to +0.01931 across 5 folds |
| O/D Kalman (two independent filters) | −0.0009 (Arm D) | Two scalar Kalman filters (one from O metrics, one from D) replacing the composite `kalman_filtered_stats`. Correlation with composite is only 0.775 — independent filters lose cross-metric information between O and D measurements. Recovers half the value at best |
| `kf_market_value_x_waived` | -0.00051 (t = -0.32), three-seed screen | Tested 2026-09-29. A tree splitting on `is_waived` already has every product with it, and the four overpriced waived stars sit in a region with too few rows to split. The interaction leaves them unchanged. See `docs/briefs/2026-09-29-waiver-term.RESULT.md`. |
| Partially linear waiver term (`beta * is_waived * kf_market_value`) | +0.00114 (t = +0.19), 10 seeds | Not adopted 2026-09-29 (v6.0.3). It discounts plain waivers, which owe nothing and are already priced without bias. The tree compensates on high-value players at the floor, and the max push re-inflates Lillard. The direction stays open as a separate branch for money-owed waivers. |
| `mpg × impact` interactions (4 forms) | −0.0018 to −0.0008 (t −1.00 to −0.49) | Tested 2026-08-29 on the hypothesis that minutes inflate the price of high-volume, low-efficiency players. Four forms — `mpg × composite impact z`, `mpg ×` each of the three metrics separately, `mpg × availability_3yr × composite` (a minute-weighted "total value" term), and the composite impact z on its own — every one negative on the pooled selection pool. It also **fails in the zone it targets**: on high-mpg / sub-average-impact rows (n=89) MAE moves $4.61M → $4.65M, and on the narrower volume-scorer slice (high mpg, high usage, impact < 0.25, n=47) $5.91M → $6.02M. The premise does not hold either — that zone's OOF bias is $+0.05M, and the volume-scorer slice is *under*-priced by $0.73M. Its problem is spread, not level: MAE $5.91M against $3.19M for everyone else, the bimodal `y | x` documented above. A gradient-boosted model already represents `mpg × metric` by splitting on one and then the other, so the explicit product only adds a collinear column. Arm: baseline XGBoost on the 20 base features, GroupKFold, 10 seeds, paired by fold. |

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

**This is not contradicted by the v4.4.0 signing offset**, and the distinction is
the whole reason that component sits in Stage 3 rather than in the feature list.
A feature is an *input*: `P(mechanism | x)` is a function of `x`, the trees
already extract it, and it costs 0.0073. The Stage-3 offset is an *output
correction* keyed on the REALISED mechanism -- a told parameter, in the same
class as the extension raise cap -- and it corrects a bias the features
demonstrably cannot see. It is also restricted to the four mechanisms that are
not determined by the contract value; see "Stage 3, signing component" below.

## Model — Three-Stage Pipeline

The model is now explicitly three stages. Composition lives in
`src/model/stages.py` and all three consumers (the evaluation suite, the web
export, and `predict.py`) call it. Every layer beyond Stage 2 is opt-in through
a keyword argument that defaults to off, so a consumer that has not opted in is
bit-identical to the composition it had before that layer existed — the web
export has not yet opted into the v4.4.0 signing offset (ISSUES #46):

```
latent  ->  push  ->  clip(lo, hi)  ->  stage 3  ->  stage 3 signing
          \________  stage 2  _______/     \___ told-route components ___/
```

- **Stage 1** estimates value under *default parameters* — what the market pays
  a player with these characteristics, with signing context averaged over the
  training distribution rather than fixed at any particular value.
- **Stage 2** adjusts for *told parameters* — the CBA bounds that apply to the
  player himself: his max tier and the league minimum. The push lives here
  because it is the upward half of the same bound: the clip alone can only cap
  from above and cannot reach a max-worthy player the model prices below his
  ceiling. The push moves the latent toward the ceiling where the route
  classifier says P(max) >= 0.52, and the clip then caps the result into
  `[floor_pct, max_eligible_pct]`. Deterministic, needs no route.
- **Stage 3** adjusts for what only applies once the **signing route** is
  known, in two components. The extension raise cap (v4.0.0) only ever lowers,
  is a no-op where `is_extension` is false or `ext_cap_pct` is NaN, and never
  reads the target. The signing-type offset (v4.4.0) adds one constant per
  eligibility mechanism and re-applies both legal bounds afterwards.

Everything else (market conditions, negotiating posture) stays averaged inside
Stage 1, which is precisely what the C2 mechanism-bias table measures.
**Promoting a parameter from "averaged" to "told" should shrink its C2 bias** —
that is the natural acceptance test for any future extension of the pipeline,
and it is the test the v4.4.0 signing offset was written to pass.

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
    objective=make_tobit_obj(right_mask, sigma=0.02, left_mask=left_mask),
)
```

Standard XGBoost treats all training rows equally. But a salary pinned against a
CBA bound is not a market price — it is the bound. **Grabit** (Sigrist &
Hirnschall, 2019) replaces the loss with a censored-normal likelihood, and since
v3.1.0 the project censors **both** bounds:

- **Uncensored rows**: standard squared error. `grad = pred − actual`, `hess = 1`.
- **Right-censored rows** (max contracts): the observation is a *floor* of the
  latent — a player worth 40% of the cap is paid 35% because that is the maximum.
  `grad = −σ·m(z)`, `hess = m(z)·(z + m(z))` with `m(z) = φ(z)/Φ(z)` the inverse
  Mills ratio and `z = (pred − actual)/σ`. Pushes predictions *above* the
  observation.
- **Left-censored rows** (at the veteran minimum): the observation is a *ceiling*
  of the latent — the league floor props the player's pay above his unconstrained
  price. Mirrored: `z = (actual − pred)/σ`, `grad = +σ·m(z)`. Lets predictions
  fall *below* the observation.

The model outputs a **latent value** — what the player would earn in a market
with no CBA bounds at all.

**Gated censoring** on both sides. Not every row at a bound is bound-constrained:

- *Right gate*: albatross contracts (John Wall 2020, Gordon Hayward 2023) are
  paid the max despite declining performance, and censoring them would corrupt
  the signal. Censor only where the baseline XGBoost prediction ≥ 55% of
  `max_eligible_pct` — about five rows filtered per run.
- *Left gate*: a ring-chasing veteran on a minimum is making a **choice**, not
  hitting a constraint, and his latent should stay meaningful. Censor only where
  the baseline prediction ≤ `k × observed`, with **k = 2.0** chosen by screening
  {1.5, 2.0, 3.0} on the floor zone.

Both gates encode the same rule: **the model must corroborate that the bound
binds.**

#### Why the floor is censorable and "good players on minimums" is not

METHODOLOGY previously recorded that extending censoring to the lower bound was
a dead end. That finding stands *for the population it tested* — treating good
players on minimums as right-censored — and the reason is economic, not
statistical: a good player on a minimum could have signed elsewhere for more, so
his salary reflects a **choice** and the censoring premise is simply false. The
oracle experiment (fold-honest `P(mechanism | x)` scores −0.0073; a leaky oracle
gains only +0.0137) measured that dead end.

The v3.1.0 population is the opposite one: players whose unconstrained price sits
*below* the minimum, held up by a rule **no contract can cross**. That is a
genuine constraint, mathematically identical to the max ceiling with the sign
flipped, and it was worth $0.42M per row on 297 rows.

### Stage 2: CBA bounds (push + clip)

```
pushed = latent + P(max) * (MARGIN * ceiling - latent)   where P(max) >= TAU
final  = clip(pushed, floor_pct, max_eligible_pct)
```

TAU = 0.52 and MARGIN = 1.05 are pre-registered constants in `stages.py`. TAU
was chosen from the expected-win-minus-expected-collateral rule before any score
on the arm was seen; MARGIN was frozen since route-mixture phase 1 and must not
be tuned on zone MAE (the clip makes censored sides one-way valves, so zone MAE
is monotone in the margin).

#### The route classifier's inputs are curated, not inherited (v4.3.0)

The classifier that produces P(max) reads
`route_mixture.CLF_BASE_COLS + CLF_EXTRA_COLS` — 36 columns, **stated
explicitly and frozen**. Until v4.3.0 the base half was `list(FEATURE_COLS)`, so
every regression feature was force-fed to the classifier as well. That coupling
is not neutral, because P(max) feeds a knife-edge gate: a column carrying no
route information still perturbs the classifier's trees through
`colsample_bytree=0.8`, and rows near TAU flip in and out of the push at random.

Measured with the 17th regression feature force-fed (5 folds × 10 seeds), the
classifier does not get *worse* — max-class AUC 0.9828 → 0.9830, multiclass
log-loss 0.7604 → 0.7539, mean |ΔP| 0.003 — which is precisely the trap: the
damage is not in classification quality but in **26 (row, seed) gate decisions
flipping on 10 borderline rows** (max |ΔP| 0.154), two of them flipping even on
the seed average. Chet Holmgren 2026 gains a push, Jimmy Butler 2023 loses one,
Jamal Murray 2025 — a true max paid *at* his ceiling — loses his in one seed of
ten, and Sabonis 2020, paid $19.80M against a $27.29M ceiling, gains one in two.
The regression's paired selection delta pays −0.00215 (t −1.25) for it,
concentrated in a single fold, which was enough to drop `playoff_mpg_diff`'s
gate from t 2.71 to t 1.51.

`CLF_EXTRA_COLS` was curated for route-discriminating signal (the
failed-regression batch) and the base list is now curated the same way. A new
column joins the classifier only by a deliberate edit to `CLF_BASE_COLS`, which
is a change to a shared contract — `deployed_p_max`, `predict.py`,
`export_web.py` and the suite all route through it — and wants its own version
number and a bit-identity check.

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
offline into `salaries_prehistory.csv` (767 rows, ~550 players, 2016–2018; the
2018 count grew from 155 to ~550 via the v3.1.2 team-salary scrape).

#### Why the ceiling is audited

A ceiling below observed pay is provably wrong — the salary happened, so it was
legal — and it does damage twice: the row is mislabelled as censored at a
threshold it has already passed, and the Stage-2 clip pins the prediction below
the truth, a guaranteed error. Auditing `actual > max_eligible_pct` found 13 such
rows before v3.0.3 (Curry 2019 at 36.9% of a frozen cap against a 35% ceiling,
Wall and Towns on early-signed designated-veteran deals, and two rows that were
float dust at exactly the tier). All three rules above came from that audit; the
count is now **zero**, and it is worth re-running after any change to experience,
awards, or cap data.

#### The designated-ceiling gate (v3.2.1)

The award path in `_compute_max_eligible` granted the 35%/30% designated ceiling
from All-NBA + experience alone, without the CBA's team-continuity requirement.
12 genuine maxes were mis-tiered: players who changed teams (Kawhi 2019), were
acquired on veteran deals (AD 2020), or signed the 25% base of a rookie
extension whose Rose escalator lands in a later year (KAT 2019, Tatum 2021). A
curated `designated_ineligible.csv` — the inverse of `early_supermax.csv` —
gates the award loop. Both curated lists share the same data gap (no team signal
in the frame) and the same eventual fix (a signing-date team-match test).

#### The Stage-2 clip never reads the observed salary (standing decision, 2026-07-26)

Clipping DOWN at actual pay is straight target leakage: over-prediction becomes
impossible, so every bargain vanishes from the Value Board by construction and a
model that predicts $40M for everyone scores well. Clipping UP to actual pay
when the ceiling came out too low is logically defensible (pay ≤ legal max, so
the salary certifies a lower bound on the true ceiling) but is rejected for two
reasons: (a) that contradiction is currently the project's most sensitive
data-error detector — it is how ISSUES #19's 12 mis-tiered rows were found —
and auto-repairing it silences the alarm, the same failure shape as the
stale-cap/stale-salary lockstep (where wrong cap ÷ wrong salary agrees and the
cross-check passes); (b) `predict.py` has no salary for an unsigned free agent,
so the guard cannot run at inference and CV would drift optimistic relative to
deployment. **Correct form: assert loudly, fix the input.** Empirically the
situation is already zero — the only rows above their ceiling are three
float-dust cases at ratio 1.0000 (Adebayo/Tatum 2021 at 0.250000, Giannis at
0.350001), so any over-ceiling check should use the same 1e-4 tolerance
`_filter_mislabeled_year1` uses.

#### The floor

`_compute_floor` is the ceiling's mirror. `is_at_floor` marks Minimum-labelled
rows inside the veteran-minimum band (≤2.5% of cap; the prorated filter has
already removed everything below 1.2%). `floor_pct` is a
(season, experience-bucket) lookup whose values are **recovered from the data's
own mass points** rather than from maintained CBA tables — 0.0148 of the cap for
two-year veterans, 0.0235 for ten-year veterans, visible as spikes in the target
distribution. Buckets are 0-2 / 3-5 / 6-9 / 10+ years of experience, falling back
to the season minimum where a cell is empty.

The `is_at_floor` label reads the observed outcome and is therefore a
**training-time** device, exactly as `is_max_contract` is on the other side. At
inference the floor clip needs only season and experience, both knowable before
the market opens.

### Stage 3: the extension raise cap (v4.0.0)

A first-paying-year extension additionally faces its own raise cap: 120% of
the player's prior-year salary under the 2017 CBA, 140% under the 2023 CBA,
**or the same multiple of the league's published Estimated Average Player
Salary, whichever is greater**. That number is usually far below the tier
ceiling, so the push can send an extension row to a ceiling it could not
legally reach. Stage 3 returns it to the law.

Three distinctions carry the whole result, and each was a way an earlier
measurement failed:

1. **The cap governs only the first paying year.** A renegotiated season
   covered by an older extension span is excluded (`span_start == season`).
2. **Rookie-scale extensions are capped by the tier instead**, classified by
   whether the previous season was a rookie-scale season and never by a text
   keyword.
3. **Designated Veteran extensions are exempt.** Eligibility is tested at both
   the signing and paying season because each catches cases the other misses.

The multiplier keys on the **signing** season: seven rows sit on exactly 1.200
and were all signed by 2022, six sit on exactly 1.400 and were all signed from
2023, with zero crossings. This differs from `cba_era`'s boundary and both
are correct -- `cba_era` governs which CBA a season is *played under*; the
raise-cap multiple governs which CBA the *deal was signed under*.

`ext_cap_pct` never enters `max_eligible_pct` or the Stage-1 censor mask: the
raise cap binds only conditional on choosing to extend, which is the same
"choice, not constraint" distinction that killed right-censoring good players
on minimums.

Evidence: `docs/briefs/2026-07-26-extension-route.RESULT.md`,
`docs/briefs/2026-07-27-told-clip-and-data-fix.RESULT.md`.

#### The told-route convention

Adopted 2026-07-26 and refined 2026-07-27. Stage 3 reads the realized signing
route, so numbers computed with it are **told-route numbers**. They go in the
**same column** as ex-ante numbers: the route is information the champion also
had and merely ignored. **v3.0.0-v3.2.2 predate the convention** and every entry
from v4.0.0 says so.

The refinement is a per-route test that keeps it honest: *after being told the
route, does the salary still require a non-trivial computation?*

- **Extension -- counts.** Told "he extended", you still compute 1.40 x prior
  pay (or the average-salary alternative) to land on Brunson's $34.94M.
- **Floor -- does not count.** `floor_pct` sits within $0.13M of observed pay,
  so being told the route is being told the answer. The floor branch's told
  arm scored +0.046 and it measures recitation.
- **Signing mechanism -- counts, for the four eligibility mechanisms.** Told
  "he re-signed on Bird Rights", you know only that his own team held his
  rights; the price is still the whole problem, and the offset moves the row by
  $1.25M on average against a $3.7M MAE. It does *not* count for the exception
  mechanisms, which is the same judgement as the leakage ruling below reached
  from the other direction: told "he signed the MLE" you have been told the
  number.

### Stage 3, signing component: the per-type residual offset (v4.4.0)

The stack carries a systematic Signing Residual by mechanism that no feature
removes -- Bird Rights rows underpriced by $1.86M, Non-Bird overpriced by
$1.22M (selection pool, v4.3.1 champion OOF). The correction is rung 1 of the
obvious ladder and deliberately the crudest thing that removes it: **one
constant per type**, added to the composed prediction.

```
pred_corrected = clip( minimum( minimum( pred + offset(type), mech_cap ),
                                ext_cap ), floor, ceiling )
offset(type)   = n / (n + k) x mean OOF residual of that type,  k = 20
residual       = actual - predicted, in cap_pct
```

The full chain order in `compose()`:

```
latent -> push -> clip[lo,hi] -> signing offset -> mechanism cap clip
-> extension clip -> re-clip[lo,hi]
```

Legality is re-applied after the offset, in that order, for the same reason
Stage 3 exists at all: a correction that pushed a row past its ceiling would be
pricing an impossible contract. On the 873-row frame it clips 54 of the 438
eligible rows back to where they started.

**Eligible types, and why the list is short.** Only the four ELIGIBILITY
mechanisms are corrected: **Bird Rights, Cap Space, Early Bird, Non-Bird**.
Every other label -- MLE, BAE, Minimum, Rookie Scale, Other, Unknown -- takes a
zero offset and comes out bit-identical, which is asserted by the suite rather
than assumed.

This is a **leakage ruling**, decided before any score on the arm was seen, and
it is the same rule as "never feed the model anything derived from the target".
An exception mechanism is *determined by the contract value itself*: a deal is
"the MLE" because of what it pays, "a minimum" because of what it pays. The
label is downstream of the target, so conditioning a prediction on it reads the
target. The four eligibility mechanisms are determined by the player's prior
contract and the team's books, both settled before the price is. The four stay
in however large the excluded types' biases look -- and Minimum's +$1.94M is
the largest on the board -- and the excluded types stay out however large
theirs look.

**Sign & Trade (and Extend & Trade) is reclassified as Bird Rights, not
excluded as leakage (2026-08-07).** It was originally grouped with the
exception mechanisms above, but the reasoning that keeps MLE/BAE/Minimum out
does not hold for it: a sign-and-trade's dollar value is not determined by the
mechanism itself -- contracts using it range $3.6M-$37.2M in the data -- and
the CBA requires the ORIGINATING team to hold Bird or Early Bird rights on the
player for the trade to be permissible at all. That makes the label a fact
about eligibility settled before the price is, the same as the four types
above, not a restatement of the price. The raw Spotrac labels
`sign-and-trade` and `extend-and-trade` now map to the `Bird Rights` category
at the signing_cat layer (`scripts/diagnostics.py::_categorize_signing`) and
take the Bird Rights offset like every other row in that bucket.

**k = 20 is pre-registered.** It lives in `stages.SIGNING_K` beside TAU and
MARGIN and is never swept. A k = 0 arm (raw per-type means, no shrinkage) was
computed REFERENCE-ONLY and decided nothing; it scores slightly better on MAE
(+$0.219M against +$0.201M) and slightly worse on the confirmation canary, which
is exactly the pattern shrinkage exists to insure against on an n = 14 type.

**Fold honesty.** An offset is a fitted parameter, so it must never be estimated
on the rows it corrects:

- **Layer A** corrects fold *f* with offsets estimated from the OOF residuals of
  rows **outside fold** *f* (leave-fold-out per-type means over the seed-averaged
  OOF). The correction is applied to each (fold, seed) cell before the seed
  average, because both the offset and the clips are non-linear.
- **Layer B** learns the offsets for target season *T* from an **inner
  GroupKFold OOF inside the training window** -- seasons < *T* only. Nothing
  from season *T* enters the offset in any capacity.
- Offsets are estimated over **all** rows including the confirmation split,
  because confirmation rows already sit in every training fold the champion
  sees; they are excluded from the deciding metric, not from the estimate
  (ISSUES #20a).

One residual channel, stated rather than hidden: layer A's leave-fold-out offset
for fold *f* averages OOF residuals of rows in folds *g != f*, and each of those
predictions came from a model that had fold *f* in its training set, so fold
*f*'s targets touch the offset through the regression's parameters. That is the
ordinary single-level-CV channel every hyperparameter chosen on OOF already
pays; a fully nested design would cost 5x the fits for a two-parameter
statistic. Layer B is clean of it entirely, and layer B is where the gain is
largest.

**Deployed constants are data-dependent.** `stages.SIGNING_OFFSETS_DEPLOYED`
holds the four numbers as measured on the 873-row frame (regenerated 2026-08-07
from the 896-row values), for the single-fit consumers that have no OOF to
estimate from. They are per-type mean residuals of a particular frame, so a
training-data rebuild invalidates them exactly the way it invalidates a
published R2. Regenerate with `scripts/eval_stage3_signing.py` and copy
`deployed_offsets_k20` across.

**`predict.py` is untouched by this.** An unsigned free agent has no signing
mechanism -- the label exists only once the contract does -- so every offset on
that path is 0.0 and the free-agent valuations are bit-identical to v4.3.1. The
deployed offsets are passed in explicitly and the identity is asserted, so the
no-op is demonstrated by the code rather than assumed.

### Stage 3, mechanism cap: the CBA ceiling for Early Bird and Non-Bird (v4.5.0)

A zero-parameter deterministic clip, the same architecture as the extension
raise cap: it only ever LOWERS a prediction, reads no fitted parameter, and the
numbers come from the CBA's published minimum salary scale and the league's
Estimated Average Player Salary.

```
Early Bird cap = max(1.75 x prior_salary, 1.05 x EAS)
Non-Bird cap   = max(1.20 x prior_salary, 1.20 x vet_min_for_experience)
```

The mechanism cap sits AFTER the signing offset and BEFORE the extension clip in
the chain, so a positive offset that pushes a row above its mechanism ceiling is
caught before the extension raise cap, which may be tighter still, governs last.
For all signing types other than Early Bird and Non-Bird, `mech_cap_pct` is NaN
and the clip is a no-op.

On the 873-row frame: 65 mechanism caps computed (52 Early Bird, 13 Non-Bird, 1
excluded for a documented label issue: josh okogie 2023). Zero violations
against actual salary, which means the CBA scale tables and formulas are
correctly calibrated to the data.

The mechanism cap is NOT the same quantity as the signing offset. The offset is a
fitted parameter (per-type mean OOF residual); the mechanism cap is a legal
ceiling. The offset adjusts the model's systematic bias; the cap enforces the
law. They are composed, not alternatives.

One exclusion, preserved by design: **josh okogie 2023** carries a Non-Bird
label inconsistent with the 2017 CBA minimum scale for 5 years of service.
120% of his vet minimum is $2,319,523 but his actual salary is $2,816,000 -- a
$497K gap that no experience count closes. Flagged as a wrong label; excluded
from the mechanism cap computation.

The vet minimum scale is implemented from two sources:
- **2017 CBA (seasons 2019-2022)**: Exhibit I published figures, the 2019-20
  base frozen through 2020-21 per the COVID agreement, then 3.0% and 6.1%
  annual increases.
- **2023 CBA (seasons 2023+)**: tier ratios derived from the data plus the
  Non-Bird ceiling rows, with a base percentage of 1.4848% of the cap.

Evidence: `src/model/mechanism_cap.py`, `scripts/eval_stage3_model.py`.

**Rung 2, tested next, not implemented.** The obvious next step is a per-type
*slope* rather than a per-type constant -- `pred + a_t + b_t x pred`, or
equivalently a per-type calibration line -- which would let Bird Rights be
underpriced by a fraction of value rather than by a flat $1.9M. It is not in the
model. It doubles the fitted parameters on types as small as n = 14, it
interacts with the clips in a way a constant does not (a slope can move a row
across its ceiling from below), and rung 1 already removes most of the
mechanism bias. Measure it against rung 1 on the eligible-type MAE, paired by
fold, before proposing it.

Evidence: `scripts/eval_stage3_signing.py`,
`outputs/models/stage3_signing_offset_eval.json`.

### Hyperparameters

Base XGBoost hyperparameters found via two-phase grid search (Phase 1: depth×n_est×lr, 64 combos; Phase 2: mcw×sub×col, 27 combos; 10 seeds each).

Tobit-specific:
- **σ = 0.02**: controls the balance between censored and uncensored gradients. CV-optimal across grid [0.02, 0.04, 0.06, 0.10]. Larger σ improves holdout but risks overfitting. Shared by both sides.
- **Right gate = 0.55**: baseline prediction must reach 55% of `max_eligible_pct` to be censored.
- **Left gate k = 2.0**: baseline prediction must be at most 2.0 × observed pay to be censored. Screened over {1.5, 2.0, 3.0}; 3.0 improved the floor zone slightly more but pushed the sub-2% predicted band further negative, and 2.0 was the best floor-zone gain that left non-floor rows untouched.

**Re-swept at v3.1.0 (2026-07-23); the incumbent held.** A 36-config grid
(sigma × gate_frac × k_floor) plus an 11-config boundary probe confirmed that
0.02 / 0.55 / 2.0 is optimal — or rather, that the alternatives fail for a
structural reason rather than by a close margin.

**Never select sigma on zone MAE.** Both censored sides are one-way valves:
every zone row is biased toward its CBA bound, and Stage 2's clip makes
overshooting free, so zone MAE falls monotonically in sigma out to 0.06 with no
interior optimum. The pinned-at-bound share climbs 19% → 42% in step with
sigma, and the cost lands on the calibration slope (0.9883 → 0.9647 at
σ = 0.04). Selecting sigma this way degenerates into "how many rows do you want
pinned to the bound" — the clip's geometry is choosing the parameter, not the
data. Evidence: `docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md`.

**gate_frac is inert on this row set** (52–56 of 57 max-zone rows gated across
0.45–0.65); the binding constraint is `is_max_contract`'s 0.90 threshold, not
gate_frac. k_floor is monotone and saturates just past 3.0 — its two admissible
settings (2.5, 3.0) clear every guardrail but their floor-zone gains ($0.061M,
$0.090M) fall short of the brief's $0.10M bar.

`_make_tobit_obj` also carries an inert `sigma_left` hook (bit-identical when
unset); the per-side control it enabled showed the two censored sides are
independent and additive, confirming the valve diagnosis.

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

- **Fixed evaluation set** whenever the training filter changes. R²'s denominator moves with the row set, so R² across different datasets is not comparable. The apparent v1.10.0 → v1.10.1 collapse from 0.866 to 0.645 is this effect, not a regression.
- **Baseline ladder**, so absolute R² is not mistaken for skill: predicting the mean 0.000, `mpg` alone **0.5740**, `mpg + prev_cap_pct` 0.6179, four features 0.7456, full 14 features 0.7609. Minutes per game alone reaches 75% of the full model's R²; the remaining ten features together buy +0.015 over that four-feature model.
- **Locked confirmation split**: 15% of players by stable hash, excluded from the metric that decides since 2026-07-23. Currently 0.7857 (n=168) against 0.7616 on the selection pool. The two levels are not comparable to each other — different players, different difficulty — so only the *trend* is informative. It widened against the pool from +0.0150 (v3.0.1) to +0.0228 (v3.0.3), an audit confirmed the drift was real, and the protocol changed in response. See "The confirmation split, and why decisions now exclude it" below.

### Results (v3.1.0, 10 seeds, n = 1,172)

| Metric | Baseline XGBoost | Grabit v4 (two-sided) |
|--------|-----------------|------------------|
| A1 CV R² | 0.7633 | **0.7653** |
| A2 CV R² (2024-26, n=386) | — | **0.8465** |
| B1 forward R² (2024-26) | — | **0.8324** |
| B1 95% CI | — | [0.779, 0.877] |
| CV MAE | — | **$3.065M** |
| CV bias | — | −$0.116M |
| Calibration slope | — | 0.9885 |
| Spearman | — | 0.7901 |

Forward R² by origin: 2024 → 0.852 (n=132), 2025 → 0.824 (n=131),
2026 → 0.812 (n=123). The cost of the forecasting setup relative to A2 is
−0.014.

Fold sd is 0.044 against seed sd 0.0014 — a factor of 31, which is why every
comparison is paired by fold.

### Judging Grabit: the zone scorecards

**Selection-pool paired delta (Grabit − baseline): +0.0022 ± 0.0016, t = +1.39.**
This has ranged from +0.0011 (t = +2.72) at v3.0.1 to −0.0001 (t = −0.07) at
v3.0.3, when the ceiling bug that had been inflating it was fixed.

The pooled test is the wrong instrument regardless of what it reads. Censoring
touches 57 + 297 = 354 of 1,172 rows, and the two sides pull in opposite
directions, so a single pooled statistic averages a large max-side effect and a
large floor-side effect over rows that neither touches. The suite reports a
**scorecard per zone**:

| Max zone (n=57) — paid ≥90% of own ceiling | Baseline | Grabit |
|---|---|---|
| MAE | $7.07M | **$6.11M** |
| bias | −$6.60M | −$6.07M |
| rows better / worse | — | **55 / 2** |

| Floor zone (n=297) — pinned at the CBA minimum | One-sided (v3.0.6) | Two-sided (v3.1.0) |
|---|---|---|
| MAE | $2.40M | **$1.98M** |
| bias | +$2.37M | +$1.93M |
| rows better / worse | — | **254 / 43** |
| per-row \|error\| change | — | **−$0.42M, 95% cluster CI [−0.469, −0.374]** |

Bias stays near −$6M in the max zone by construction, not by failure: those
players are paid their ceiling and Stage 2 caps predictions at that ceiling, so
the residual can only be ≤ 0. MAE is the number that moves. The floor zone is the
mirror image with bias ≥ 0.

**The floor side also moves the pooled dollar metric**: MAE $3.181M → $3.065M,
95% cluster-bootstrap CI [−0.139, −0.091], excluding zero. R² barely notices the
same change (0.7647 → 0.7653) because it weights by squared error and these are
297 small-dollar rows, while MAE weights by row and they are 25% of the sample.
**Where R² and MAE disagree this sharply, the disagreement is arithmetic, not
contradiction** — read both.

**Spillover differs by side, and one set of trees serves every row.** On the max
side it is two-sided: rows at 75–90% of their ceiling are not censored yet gain
(bias −$3.21M → −$2.74M), while rows at 50–75% pay for it (MAE +$0.13M). On the
floor side it is favourable — the neighbours were overpredicted too, so the
downward pull helps them: non-floor rows move −$0.012M with a CI spanning zero,
and the worst fixed-row segment |bias| growth is +$0.07M against the $0.30M gate.

**Segment checks between two models must use fixed rows.** A predicted-band
decomposition of v3.1.0 first read "−$1.21M on 2-4% others", which looked like
serious damage and was band-composition shift — the same regression-to-the-mean
artifact layer C exists to avoid, reappearing on the model-comparison axis. Rows
must be assigned to segments by something neither model produced.

### Acceptance rules

**Challenger changes** — features, filters, hyperparameters, anything acting on
every row — are accepted when all five hold:

1. paired A1 delta > 0 with |Δ| / SE > 2, **computed on selection-pool rows only**
2. A2 moves the same direction (significance not required)
3. no C2 segment's bias worsens by more than $0.3M
4. C1 calibration: the candidate's |slope − 1| may not exceed the incumbent's by
   more than **0.005** (≈ $0.3M of scale distortion at a ~$60M max — the same
   materiality threshold C2 uses)
5. MAE does not regress beyond the agreed tolerance

**The C1 gate is relative, not absolute** (adjudicated 2026-07-23). The earlier
absolute window [0.99, 1.01] excluded the champion's own slope (0.9883) — a
drafting error discovered during the sigma/gate retune. The relative gate uses
C2's own $0.3M yardstick: 0.005 of scale distortion on a ~$60M max. Paired
slope-difference noise is far below 0.005, so this is a real tolerance, not a
noise band.

**Pure-removal changes** (filter tightening that drops rows without altering the
model) cannot be judged on common-row A1, which scores only rows both frames keep
and is by construction blind to the benefit of removing stale rows. The judging
criteria are: (i) the pre-registered contamination signal shrinks or collapses;
(ii) C2 fixed-segment |bias| growth stays within $0.3M; (iii) common-row A1 is
neutral (the change does no harm). v3.1.1 established this: its common-row A1
was −0.0009 (t −0.13, neutral) while its pre-registered `is2019` control
collapsed from +0.0035 (t 2.29) to −0.0004 (t −0.91).

**Each Grabit side** is judged on its own zone: keep it while that zone's MAE
delta is negative, drop the side whose zone turns positive. Applying the pooled
rule to a 5% intervention would have removed the max side at v3.0.3 on a
t-statistic of −0.07.

**The correctness-versus-metric thread.** Six of the last nine versions landed
on a correctness argument rather than a metric win, and several did not clear
t > 2: v3.1.1 (common-row A1 -0.0009), v3.2.2 (t = 0.08), v4.0.0 (t = 1.11),
v4.0.1, and the missingness repair inside v4.1.0 (t = 1.10, failing the
calibration guard by 0.0017). The standing rule these encode: **a wrong fact is
repaired on correctness; a suboptimal parameter must clear the gate.** The
protocol's role in a correctness case is to confirm the fix does no harm, not
to justify it.

**Rule 3 is unreliable for level corrections.** The guard compares |bias|, so a
change that shifts every segment by the same amount necessarily trips it wherever
a segment was already overpredicted — v3.0.1 breached it while improving four
segments and worsening two by the identical mechanism. Treat a C2 breach on a
calibration change as uninformative until the guard measures segment bias
*relative to the global level*.

### The confirmation split, and why decisions now exclude it

Every accept/reject in this project's history read the same metric. Each decision
carries noise and the winning side is kept, so across 20+ feature decisions, two
hyperparameter sweeps, and the sigma and gate thresholds, the metric can drift
upward without the model improving. The confirmation split — 15% of players,
assigned by a hash of the name so it cannot drift — is the canary: it carries
only 15% weight in a pooled decision metric, so it should climb more slowly than
the selection pool, but not fall while the pool rises.

It fell. The audit promised at v3.0.3 was run over v3.0.1 → v3.0.4, on identical
rows with identical fold assignment:

| Slice | v3.0.1 | v3.0.4 | delta |
|---|---|---|---|
| Selection pool (n=1,107) | 0.7568 | 0.7603 | **+0.0035** |
| Confirmation split (n=184) | 0.7533 | 0.7481 | **−0.0052** |

Difference-in-differences on row-level squared-error improvement, bootstrapped by
player cluster: **+5.6e-5, 95% CI [+8.5e-6, +1.11e-4] — excluding zero**, and the
result survives removing the six rows v3.0.5 later demoted.

**The protocol changed in response.** `oof_groupkfold` returns a second fold ×
seed matrix computed on selection-pool validation rows only; `paired_delta` for
accept/reject reads that matrix, and both persist in `evaluation_suite.json`
(`fold_r2_selection` decides, `fold_r2` is context). Confirmation rows still
serve as training data in other folds — they are excluded only from the metric
that decides. Headline A1/A2/B1 stay pooled for continuity.

Honest limits on the finding: the CI's lower bound sits near zero, and one
plausible channel is innocent — v3.0.2's rookie-scale fill table is estimated from
the whole dataset, 85% of which is selection-pool players, so it would help those
rows more without anyone gaming anything. The evidence is *moderate and
directionally clear*, not a verdict. Tightening costs 7% of the decision sample,
which makes it a cheap precaution either way.

The split is a canary, not a sealed envelope: the suite prints it on every run.
Per-version per-row OOF is archived in `outputs/models/oof_reference.csv`, and
the suite now rotates one generation to `oof_reference_prev.csv` before
overwriting, so the paired comparison against the previous state survives a
rerun.

Post-change readings (v3.1.0): selection 0.7616, confirmation 0.7857 — the
confirmation slice now scores *above* the pool, which is what the v3.0.5-v3.0.6
cleanups look like when they help rows nobody was watching.

## Holdout vs Valuation

Two distinct scoring modes, and CONTEXT.md gives them separate names because
they mean different things:

- **Signing Board / holdout**: Year-1 filter on train and test both. Produces the **Signing Residual** — a measure of model accuracy against a price the market actually set.
- **Value Board / valuation**: Year-1 filter on train only, score every row. Produces the **Contract Surplus** — a statement about a team's books, not about model error. R² is not meaningful here; ranking is.

## Diagnostic Findings (v3.1.0, n = 1,172)

### Bias by predicted band

| Predicted band | n | bias | MAE |
|---------------|---|------|-----|
| <2% | 301 | −$0.42M | — |
| 2-4% | 261 | −$0.17M | — |
| 4-8% | 286 | +$0.34M | — |
| 8-15% | 185 | −$0.13M | — |
| 15-25% | 96 | −$0.15M | — |
| 25%+ | 43 | −$0.51M | — |

Flat to within ±$0.51M — tighter than at v3.0.3 (±$1.05M), mostly because
v3.1.0's floor clip removed the overprediction that used to sit in the bottom
bands. Calibration slope 0.9885, intercept +0.0019.

**Bands must be cut on the prediction, never on the target.** Binning residuals
by actual salary produces a steep monotone gradient even when calibration is
perfect; earlier versions of this document reported that artifact as a finding.
The same trap reappears when *comparing two models* — see the band-composition
warning under "Judging Grabit".

### Signing Residual by mechanism

| Mechanism | n | bias | MAE |
|-----------|---|------|-----|
| Bird Rights | 309 | −$2.50M | $4.76M |
| Minimum | 303 | +$1.86M | $1.99M |
| MLE | 182 | +$1.05M | $2.26M |
| Cap Space | 117 | −$1.18M | $3.60M |
| Unknown | 132 | +$0.24M | $2.56M |
| Early Bird | 48 | −$1.35M | $2.74M |
| Other | 41 | +$1.45M | $2.03M |
| Non-Bird | 22 | +$1.92M | $2.38M |
| Sign & Trade | 15 | −$3.83M | $4.48M |

Minimum's bias fell from +$2.51M to +$1.86M and its MAE from $2.59M to $1.99M
between v3.0.3 and v3.1.0 — that segment is the floor zone under another name, and
the left-censoring branch is what moved it.

Bird Rights and Sign & Trade are underpriced — both let a team exceed the cap for
its own player, and the retention premium is invisible to the features. Minimum
and MLE are overpriced. These patterns survive controlling for predicted value
but are still not usable as features; see "Why signing mechanism cannot be a
feature."

*(Historical note, 2026-08-07: this table predates the decision to fold Sign &
Trade -- and Extend & Trade -- into Bird Rights (see "Stage 3, signing
component" above). Its 15 Sign & Trade rows are Bird Rights rows under the
current signing_cat categorization; re-running this diagnostic today would
show them merged into the Bird Rights row rather than listed separately.)*

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

### Grabit impact in its zones

See "Judging Grabit: the zone scorecards" above. Over the 57 rows paid ≥90% of
their own ceiling, MAE falls $7.07M → $6.11M (55 better, 2 worse). Over the 297
rows pinned at the CBA minimum, MAE falls $2.40M → $1.98M (254 better, 43 worse),
per-row |error| −$0.42M with a cluster-bootstrap CI of [−0.469, −0.374].

## Inference Pipeline

`src/model/predict.py`:
1. Load training data, apply the full filter chain (year-1, rookie, prorated,
   mislabel, continuation)
2. Compute `max_eligible_pct` (Rose Rule / supermax / no-decrease floor) and
   `floor_pct` (season × experience bucket)
3. Train baseline XGBoost → compute both gated censoring masks
4. Train Grabit XGBoost (two-sided censored loss) on all training data
5. Predict latent value → apply CBA bounds: `clip(latent, floor_pct, max_eligible_pct)`
6. Convert cap_pct to salary dollars
7. Output: predictions CSV + free agents CSV

Both bounds are knowable before the market opens: the ceiling needs experience
and award history, the floor needs season and experience. The `is_at_floor` and
`is_max_contract` labels read observed pay and are training-time devices only.

## Known Issues & Limitations

### Modelling

1. **Rookie max extensions**: players like JJJ, Jalen Williams get max extensions based on projected ceiling — the model sees current-season stats only, not future potential.
2. **Albatross Contracts**: John Wall, Gordon Hayward — paid max despite poor performance. Gated censoring filters most but not all. The model correctly says they're overpaid.
3. **Signing mechanism is a label, not a feature** — partly determined by the contract itself, and empirically worthless as a modelled probability. See the dedicated section above.
4. **No tracking data**: NBA.com tracking data (drives, catch-and-shoot, rim protection) could improve archetype-specific predictions.
5. **σ tension**: CV prefers σ=0.02 (conservative censored gradients), holdout prefers larger σ (more aggressive). Sticking with CV-optimal to avoid overfitting to a single holdout season.
6. **Point estimates against a bimodal target**: `y | x` is genuinely bimodal for mid-market players. A conditional mean is the R²-optimal point estimate and is nonetheless wrong for both modes. A distributional output — quantiles, or `P(signs for the minimum)` alongside a conditional market value — would answer the actual question better, and would not be measurable by R².
7. **The model is not underfitting**: depth 6, learning rate 0.03, and looser `min_child_weight` all score worse than the current settings (0.7648 / 0.7613 / 0.7527 against 0.7661). Added structure cannot be justified as an inductive-bias fix.
8. **Grabit's reach is not confined to its gates.** Censoring changes gradients for 354 rows, but one set of trees serves all 1,172, so leaf values shift for their neighbours too. On the max side this cuts both ways — rows at 75–90% of their ceiling gain without being censored, rows at 50–75% lose slightly. On the floor side it is favourable, because the neighbouring rows were overpredicted in the same direction. No threshold separates the effects; it is the same mechanism, and the net is positive on both sides.
9. **The floor lookup is a data-derived approximation.** `floor_pct` is the median pay of at-floor rows per (season, experience bucket), not the CBA minimum scale itself. It recovers the mass points closely and needs no maintenance, but a season with few at-floor rows in a bucket falls back to the season minimum. A published scale would be exact.

### Pipeline and reproducibility

10. **The contract-structure script was never committed.** Only `contract_structure_v2.csv` survives. Reconstructing the detection from CBA escalator ratios reaches at best 88% agreement on the year-1 flag and 74% on contract length across a 30-point parameter sweep — far too low to regenerate history without invalidating every published version number. `scripts/extend_contract_structure.py` therefore extends the table incrementally: rows whose salary is unchanged keep their assignment byte-for-byte, only new or changed rows are assigned, and it hard-fails if an unchanged row would move. Any future change needing a full recompute will still hit this wall.
11. **`prev_cap_pct` is produced by an experiment script.** The regeneration chain is `src/features/build_dataset.py` → `scripts/build_external_features.py` → `scripts/phase3.py::build_contract_features`, and that last stage lives in `phase3.py` for historical reasons. `scripts/rebuild_training_data.py` now chains all three behind one command and validates that the fifteen cap-independent columns reproduce exactly on shared rows, but the stage itself has not been moved to a home of its own.

## Routes Tested and Rejected

Five route-mixture architectures were measured and all five are closed. Each
tested whether a signing-route probability function applied to the Grabit
latent at **output** time (not as an input feature) could recover error the
censored model leaves on the table. The answer is no for every route except
the extension, which was adopted in v4.0.0 as Stage 3 -- and Stage 3 works
only because it applies a legal ceiling, not because a classifier learned
where the error is.

**Route mixture, full six-route form (ex ante).** dSel -0.0479, t = -2.50,
calibration slope 0.819. All five gates fail. The structural routes (max,
floor) produce route values `V_max` and `V_floor` that sit far from `f(x)` on
every row, and composing them with an imperfect P redistributes error across
the ~93% of rows that are not candidates.

**Floor branch (ex ante).** Oracle headroom is +0.0396 (six times the max
side's) and unreachable. P(floor) is anti-ranked against the error: Spearman
= -0.611 inside the zone. The classifier's most confident quarter of the zone
captures 2.5% of the over-prediction where an error-ordered selector would
capture 78.7%. The twelve worst floor rows score **above the average non-floor
row** on minutes, all three impact metrics, usage, prior pay and awards -- the
same features that make the model overprice them make them invisible to any
classifier fed those features. At the pre-registered operating point (tau =
0.10), both ex-ante arms fail: arm A (unconditional pull) dSel -0.084 at
t = -2.48, arm B (P-weighted pull with margin) dSel +0.003 at t = +0.54,
best t anywhere on the grid +1.75.

**Offseason-injury signal.** The earlier floor-only oracle of +0.0097 did not
survive a frame-wide feature test. Cached Spotrac pages produced 11,697 dated
injury/DNP rows, but the table dates missed games rather than diagnoses and
cannot expose Lillard's April 2025 Achilles tear before his July signing. All
fixed score forms were negative in the three-seed directional screen: the best
simple event-count form was -0.00172 paired Delta selection R-squared, the
type-weighted form was -0.00173, and the full time/type/recurrence score was
-0.00239 at t = -3.20. The user rejected the feature and stopped the 10-seed
gate. Reopen only with diagnosis dates and signing-time prognosis or recovery
status, not another weighting of the same missed-game table.

**Ring-chasing discount (v6.0.2).** A pull on the Stage-2 latent for veterans
likely to be on their last contract, weighted by a retirement hazard fitted on
1997-2016 careers (AUC 0.725 against the realized outcome), failed its
pre-registered gate at t = +0.88. Oracle arms that pull on the realized last
contract fail too (t = -0.43, and t = -0.50 when restricted to earnings at or
above P75). In 2019-2023 a paid last-contract signing shows no discount
(residual -$1.37M against -$1.35M for other paid veterans), and every gain
comes from rows that sign the minimum. Reopen only when the 2024-2026 star
signings have observed horizons, and only after the `last_rich` oracle passes.
See `docs/briefs/2026-09-29-ring-chasing.RESULT.md`.

**MLE branch.** P(mle) AUC 0.71: whether a player is offered an exception
depends on the signing team's cap position, which no player feature sees.

**Ex-ante per-route delta.** The fold-honest intercept correction delta_k is
composed as pred = f(x) + Sigma P(k) * delta_k. It fails on surface lift:
mean P(bird) is 0.29 on every row, so P * delta_bird moves the whole price
surface rather than landing on bird-rights rows alone.

Evidence for these closures lives in the RESULT bundles under `docs/briefs/`:
`route-mixture.RESULT.md` (phase 1, max branch + classifier),
`route-mixture-p2.RESULT.md` (purity-gated max, enriched classifier),
`route-mixture-p3.RESULT.md` (corrected labels, four-cell battery),
`floor-branch.RESULT.md`, `extension-route.RESULT.md`, and
`route-delta.RESULT.md`.

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
version tag (`v3.0.0` … `v3.1.0`) and running `src/model/evaluate_suite.py`.

## References

- Sigrist, F. & Hirnschall, C. (2019). "Grabit: Gradient Tree-Boosted Tobit Models for Default Prediction." *Journal of Banking & Finance*. — Censored-normal likelihood as XGBoost custom objective.
