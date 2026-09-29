# Work queue

Last updated 2026-09-29 after v6.0.4 (KF anchor re-pricing).

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Improve the waiver features

Production includes `is_waived` and `mpg_x_waived`; both stay (three-seed
ablations t = -0.45 and -0.53). The partially linear term failed at v6.0.3.
See `docs/briefs/2026-09-29-waiver-term.RESULT.md` for the three failure
mechanisms. Only money-owed waivers are mispriced: 35 rows, bias +$2.52M.
The 85 plain waivers have bias -$0.13M.

- `prior_waiver_owed` (landed 2026-09-29) scans every waiver in the lookback.
  36 of the 120 waived evaluation rows owe money: 26 are in selection, 10 in
  confirmation, and 24 sit at the floor.
- `owed_branch` (2026-09-29) failed at 10 seeds. Money-owed rows dropped
  from the Grabit fit and priced as `gamma * latent` (gamma mean 0.24) gave a
  targeted t of +0.73 on waived selection rows. Money-owed rows improved (MAE
  $2.93M to $0.98M; t +1.15 over 21 players). Plain waivers rose by $0.64M on
  average (t -2.12): they had borrowed the waiver discount the tree learned
  from money-owed floor rows. Keeping money-owed rows as right-censored rows
  raised plain waivers further in a one-seed prediction check (+$0.86M).
- Pre-registered 2026-09-29, before any score, as one arm `waived_branch`:
  1. Stage 1 fits Grabit only on rows with `is_waived != 1`. `FEATURE_COLS`
     is unchanged, and `is_waived` is constant in that fit.
  2. Every waived row is priced as `gamma_g * latent`. There are two groups:
     money-owed (`prior_waiver_owed == 1`) and plain.
  3. Each gamma is a Tobit fit on that group's training rows, with at-floor
     rows left-censored, bounded to [0, 1]. A group with fewer than 10
     training rows keeps gamma = 1.
  4. P(max) = 0 on waived rows. The rest is as `owed_branch`.
  5. Run: `python scripts/eval_waiver_challengers.py --seeds 10 --full --arms
     incumbent waived_branch`.
  6. Gate: the targeted gate in `docs/worker-brief.md` on `is_waived == 1`,
     with pooled selection dSel > 0, B1 in the same direction, C2 growth
     <= $0.3M and a C1 gap <= 0.005.
  7. If it fails, record it. Future signings are a monitor after adoption,
     not a gate.
- `waived_branch` (2026-09-29) failed at 10 seeds: targeted t +1.06, and
  pooled selection dSel -0.0005. Money-owed rows improved again (MAE $2.93M
  to $1.01M). Plain waivers were flat (MAE $1.02M), and non-waived rows
  worsened (t -1.99). Every arm that changes the Stage-1 training set moves
  other rows. Plain waivers are the best-priced group (MAE $1.01M against
  $2.78M for the frame) and never improved in any arm.
- Pre-registered 2026-09-29, before any score, as one arm `owed_adjust`:
  1. Stage 1 is the incumbent fit, unchanged.
  2. Only test rows with `prior_waiver_owed == 1` are changed, to
     `gamma * latent`.
  3. Gamma is a Tobit fit (at-floor rows left-censored, bounded to [0, 1]) on
     the training money-owed rows. Their latent comes from a 4-fold
     player-grouped inner OOF of the same Grabit fit, so it is out-of-sample,
     like a test row's.
  4. P(max) = 0 on money-owed rows only. This narrows the user's earlier
     `is_waived` exclusion, so that plain waivers are untouched.
  5. Before Stage 3, every other row must be bit-identical to the incumbent.
     Stage-3 signing offsets can move slightly, because they average
     residuals that include money-owed rows.
  6. Affected rows: `prior_waiver_owed == 1`. Gate: the targeted gate in
     `docs/worker-brief.md`, with pooled selection dSel > 0, B1 in the same
     direction, C2 growth <= $0.3M and a C1 gap <= 0.005.
  7. Report two sensitivity checks next to the gate, not as the gate. The
     first drops the rows the user flags as special cases: Deandre Ayton
     2025, Marcus Smart 2025 and Marcus Smart 2026. The second drops the two
     largest-gain money-owed selection rows. The user may override a FAIL
     through an ADR. That ADR must state the gate result.
  8. Run: `python scripts/eval_waiver_challengers.py --seeds 10 --full --arms
     incumbent owed_adjust`.
- Complete when `owed_adjust` is adopted, rejected, or overridden through an
  ADR.
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
