# Work queue

Last updated 2026-09-29 after v6.0.2 (ring-chasing discount rejected).

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Improve the waiver features

Production includes `is_waived` and `mpg_x_waived`. Kemba Walker 2021 shows
that the binary signal cannot describe a value-dependent discount after a
buyout.

- The leakage audit is clean after the 2026-09-29 fallback correction (see
  `VERSION_HISTORY.md`, v6.0.1). Still open: separate an ordinary waiver from a
  buyout re-signing.
- Confirm the patched waiver columns with a full local
  `python scripts/rebuild_training_data.py`; `validate()` should pass.
- Remeasure `is_waived` and `mpg_x_waived` on the v6.0.2 frame.
- Add `kf_market_value_x_waived = is_waived * kf_market_value` as a challenger.
  Build this interaction inside each CV fold after nested KF inference.
- Report paired A1, A2, B1, C1, and C2. Report named-row effects for Kemba
  Walker 2021, Hassan Whiteside 2020 and Damian Lillard 2025. The ring-chase
  oracle's largest gains were these buyout rows (Walker +$16.1M), so the
  discount belongs here. Lillard 2025 is the second-largest
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
