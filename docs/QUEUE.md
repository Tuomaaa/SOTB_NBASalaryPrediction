# Work queue

Last updated 2026-09-29 after v6.0.4 (KF anchor re-pricing).

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Improve the waiver features

Results through v6.0.5 are in `VERSION_HISTORY.md` (v6.0.3, v6.0.4) and
`docs/briefs/2026-09-29-waiver-term.RESULT.md`. On the v6.0.5 frame, two
waived groups are mispriced; bias is prediction minus actual:

| Group | Rows | Bias | Driven by |
|---|---:|---:|---|
| Fresh money-owed waivers above the floor | 8 | +$7.2M | Lillard 2025, Walker 2021, Beal 2025 |
| Stale plain waivers above the floor | 18 | -$1.77M | Okogie 2026 (89% under-predicted) |

"Stale" means a contract was signed between the waiver and this signing.
The tree applies a near-uniform waiver discount, which is too weak for stars
and too strong for mid-priced players. Arms that change the Stage-1 training
set moved other rows. Turning stale flags off (ISSUES #59) lowered A1.

- Pre-registered 2026-09-29, before any score, as one arm
  `waiver_term_noflag`:
  1. Drop `is_waived` and `mpg_x_waived` from the regression (19 features).
     The route classifier and the KF measurement model keep their lists.
  2. `latent = GBM(x) + beta * is_waived * kf_market_value`. Beta is a Tobit
     fit on 4-fold inner OOF residuals of the same 19-feature base, with
     at-floor rows left-censored and beta bounded to [-1, 0]. It enters as
     `base_margin`.
  3. P(max) = 0 on waived rows.
  4. Run: `python scripts/eval_waiver_challengers.py --seeds 10 --full --arms
     incumbent waiver_term_noflag`.
  5. Gate, set by the user: pooled selection dSel > 0, B1 moves the same way
     (B1 change >= 0), C2 growth <= $0.3M, and C1 gap <= 0.005 must all pass.
     The targeted t on waived selection rows is reported, not required.
  6. If the arm fails, record it and do not amend it.
- Complete when `waiver_term_noflag` is adopted or rejected with paired
  metrics.
- Still open: confirm the patched waiver columns with a full local
  `python scripts/rebuild_training_data.py`.

### Fold P(max) into Stage 1

Suggested by the user on 2026-09-29, after the partially linear waiver term:
the max push is also a structural term, not a separate stage. Write a brief
after the waiver arm resolves. Complete when the brief exists.

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
- Tag v6.0.0 on `30aa82e`, v6.0.1 on `093b2e0` and v6.0.2 on the commit that
  records it, each with the headline metrics A1 0.8509 / A2 0.8559 / B1 0.8304.
- Complete when `git ls-remote --tags origin` lists `pre-tidy-2026-09-28`,
  `v5.3.2`, `v5.3.3`, `v6.0.0`, `v6.0.1` and `v6.0.2`.

### Add P(max) to the Value Board

- Display max probability and conditional value, for example
  `85% max, $46.4M if maxed`.
- Complete when the board displays it without changing model scoring.

## Recurring

- Refresh the 2027 free-agent class with the chain in `CLAUDE.md`.
- Automate team-continuity refresh as maintenance. Treat it as a model feature
  only after a new evaluation brief.
