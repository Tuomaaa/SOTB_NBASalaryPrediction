# Work queue

Last updated 2026-09-29 after v6.0.0 (repeated grouped CV).

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Decide the ring-chasing discount

`scripts/eval_ringchase_gated.py` pulls the Stage-2 latent down for gated
rows. The earlier +0.93 screen scored a pre-v5.3.0 champion (wrong KF setup,
fixed 2026-09-28). Three-seed screens on v5.3.3, with a one-sample t over
(fold, seed) cells that overstates the evidence because the cells shared folds
(replaced in v6.0.0 by the corrected `paired_delta`):

| gate | n | paired dSel t | A1 without LeBron 2026 |
|---|---:|---:|---|
| age, earnings >= P75, no ring | 50 | +0.19 | 0.8367 -> 0.8369 |
| age, earnings >= P75 | 82 | +1.30 | 0.8367 -> 0.8360 |

The earnings-only gain is LeBron James 2026 alone; 23 of 50 moved rows get
worse (Butler 2023 -$6.3M, Durant 2026 -$6.1M, Paul 2021 -$5.9M). Absolute
age cannot separate discounters from full-price veterans.

- Pre-registered 2026-09-29, amended before any result: the final arm is
  `--gate continuous`: latent - delta * w * (latent - floor), with
  w = (p_last_2y - p0) / (1 - p0) above p0 and 0 below, p0 in {0.2, 0.35,
  0.5} and delta over 0-3.0 both chosen on absolute error in pool, no
  shrinkage. The hazard is v2 (age, BPM, BPM trend, two-season minutes and
  games; fitted on veterans aged 30+ in 1997-2016; 0 below age 30). The
  amendment followed a pre-run check: without a floor and age limit, 167
  players under 32 were pulled by more than $0.5M at delta = 1 because one
  injury season read as retirement (De'Anthony Melton 2025 p = 0.71). It is
  adopted only if, at 10 seeds on the v6.0.0 frame, the
  corrected paired dSel t > 2, delta is interior in most pools, A1 without LeBron James 2026
  improves, and C2 passes. Otherwise all ring-chase shapes are rejected and
  no further shape is tried.
- Screens so far (3 seeds): the hard retire gate reached t = +2.74 with delta
  capped at 1.5, but with the cap at 3.0 delta ran to the edge in every pool,
  t fell to +1.74 and moved rows split 14 better, 14 worse. The gate mixed
  valuable veterans near the end (LeBron, Horford 2025) with declining role
  players the champion already under-prices (Covington 2023, Paul 2024).
- Next: gate on the probability that this is the player's last contract,
  estimated as P(retire within 1-2 years | age, BPM, minutes) and fitted on
  completed careers. Scrape
  `python scripts/scrape_advanced_history.py` (1997-2018) on a machine that
  Basketball Reference has not jailed; 2019-2026 alone gives every
  good veteran P(retire) near 0, LeBron included.
- Fit the hazard on seasons <= 2016 with lookahead to 2018, freeze it, add a
  `--gate retire` arm, and report the 2026 origin and A1 with and without
  LeBron 2026 plus the named rows above.
- Public reporting supports the horizon reading: the seven veteran stars with
  two or fewer seasons left at signing all took less than the market (Paul
  2025, Gasol 2020, LeBron 2026, Curry 2027, Durant 2026, Harden 2025, Horford
  2025); the four with three or more took full price (Paul 2021, Lowry 2021,
  Conley 2021, Butler 2023). This uses realized careers, so it checks the
  assumption, not the ex-ante estimate.
- Kevin Durant 2026 signed $30M under his maximum (ESPN, 2025-10), but the
  champion already priced it ($43.3M against $43.9M). The pull must not
  discount a row twice; report rows the champion already prices within $2M.
- The gate estimates the belief at signing, not the realized outcome.
  DeMar DeRozan 2024 (three years at 35) may have been expected to be his last
  deal before he played on. Select the window (P(retire within 2 or 3 years))
  inside the pool, like the age threshold.
- The belief must come from the hazard, never from a narrative fitted after
  the fact. Check it: DeRozan 2024 must score clearly above Butler 2023 and
  Paul 2021 for the reading to help.
- Forward checks, both outside the frame: Stephen Curry's September 2026
  extension (two years, $116M, year 1 $55.7M in repo season 2027, about $20M
  under his maximum; ESPN) and DeRozan's August 2026 one-year minimum with
  Denver after Sacramento waived him (AP). Score both with the frozen gate
  before the refresh adds the rows.
- Complete when the retire gate is adopted or rejected with paired metrics and
  per-row gain and damage; reject all three gates if it fails.

### Improve the waiver features

Production includes `is_waived` and `mpg_x_waived`. Kemba Walker 2021 shows
that the binary signal cannot describe a value-dependent discount after a
buyout.

- The leakage audit is clean after the 2026-09-29 fallback correction (see
  `VERSION_HISTORY.md`, v6.0.0). Still open: separate an ordinary waiver from a
  buyout re-signing.
- Confirm the patched waiver columns with a full local
  `python scripts/rebuild_training_data.py`; `validate()` should pass.
- Remeasure `is_waived` and `mpg_x_waived` on the v6.0.0 frame.
- Add `kf_market_value_x_waived = is_waived * kf_market_value` as a challenger.
  Build this interaction inside each CV fold after nested KF inference.
- Report paired A1, A2, B1, C1, and C2. Report named-row effects for Kemba
  Walker 2021 and Damian Lillard 2025. Lillard 2025 is the second-largest
  forward miss ($37.21M, 0.025 of B1 R-squared).
- Complete when the audit is clean and each challenger passes the gates or has
  a recorded rejection.

### Grabit hyperparameter tuning — PAUSED

Paused by user decision on 2026-08-31. Keep the incumbent
`sigma = 0.02`, `sigma_left = 0.02`, `right_gate = 0.55`, and
`left_gate = 2.0` until this item is explicitly resumed.

The two censored sides are biased in opposite directions on the current frame:

| zone | n | mean error | MAE |
|---|---|---|---|
| max | 68 | **-$2.53M** | $2.68M |
| middle | 553 | -$0.68M | $3.24M |
| floor | 249 | **+$1.97M** | $1.97M |

The v5.3.2 screen found that stronger settings shrink both bound-zone biases
but lose whole-frame performance. At `sigma = sigma_left = 0.04`, same-run A1
fell from 0.826685 to 0.823787 and C1 failed. The balanced 0.025/0.025 arm had
paired dSel +0.000267 at t=0.77 and also failed C1. Resume only if a new
objective or new data can distinguish ordinary waivers from buyout re-signings.

### Re-derive the pre-registered constants

`MARGIN = 1.05` has no recorded derivation — METHODOLOGY records only "frozen
since route-mixture phase 1", and it cannot be fitted on zone MAE because the
clip makes that objective monotone in it. `TAU = 0.52`, `SIGNING_K = 20` and
`floor_gate_k = 2.0` were each set once.

- Re-derive each on the migrated frame under a stated objective, or record why
  the current value stands.
- Complete when every constant has either a derivation or a written reason to
  leave it alone.

## Needs a brief

### Separate ex-ante and ex-post Stage-2 modes

- Define separate code paths and reports; add the terms to `CONTEXT.md`.
- The observed Bird Rights offset uses outcome information and cannot enter
  headline CV. Define completion checks in the brief.

## Ready

### Publish the accuracy panel

Build the public accuracy panel from
`docs/briefs/2026-09-28-public-accuracy-views.RESULT.md`.

- Headline: share of B1 signings within $0.5M, $2M, and $8M, with cap points
  as the secondary unit. Do not publish the zero-error share or the 10-point
  tier.
- Show unmodified A1 and B1 R-squared first, with the trim curve beside them.
- List the top five forward misses by name with a one-line reason each.
- Source every number from `oof_reference.csv` through
  `scripts/public_accuracy_views.py`, never from the `export_web` refit.
- Decide the destination first: the Value Board through `export_web`, or a
  static README section.
- Complete when the panel shows the three tiers, the trim curve, and the named
  misses, and its numbers match `public_accuracy_views.csv` for the current
  champion.

### Push version tags

`git ls-remote --tags origin` returns no tags. `VERSION_HISTORY.md` cites
`pre-tidy-2026-09-28`, and `CLAUDE.md` requires a tag for each version bump.

- From the machine that holds the tags, run `git push origin --tags`. Local
  tags carry legacy `vN.Mx` names; add the renumbered name beside each landed
  release (for example `v5.3.3` beside `v8.17x`) with its headline metrics.
- Tag v6.0.0 on the merged commit with its headline metrics.
- Complete when `git ls-remote --tags origin` lists `pre-tidy-2026-09-28`,
  `v5.3.2`, `v5.3.3` and `v6.0.0`.

### Add P(max) to the Value Board

- Display max probability and conditional value, for example
  `85% max, $46.4M if maxed`.
- Complete when the board displays it without changing model scoring.

## Recurring

- Refresh the 2027 free-agent class with the chain in `CLAUDE.md`.
- Automate team-continuity refresh as maintenance. Treat it as a model feature
  only after a new evaluation brief.
