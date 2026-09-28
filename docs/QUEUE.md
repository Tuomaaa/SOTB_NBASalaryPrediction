# Work queue

Last updated 2026-09-28 after v8.17x (ISSUES #55).

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Finish the Spotrac salary migration

The 2019 COVID rule and the `career_earnings.csv` and `rings_thru_prev.csv`
regeneration need no action: 104 of 127 frame rows for 2019 match BBRef with no
over-corrected row, and neither file reads `salaries.csv`.

The remaining defect is multi-amount seasons. The merge summed a season's cash
across teams, so a waived player's old contract priced his new signing (Dion
Waiters 2019 read $13.44M for a rest-of-season minimum). 26 frame rows differ
from BBRef by more than $0.5M. `build_merged_salaries.py` now has the
`signing_contract` branch: price the contract that starts that season.

- The branch was fixed against the 2026-09-28 dry-run dump in
  `outputs/experiments/merge_debug/`: 49 of 52 verified rows match. Austin
  Rivers 2020 reads the page cell sum $3,476,027 against a remembered $3.5M;
  Oshae Brissett and Bol Bol 2023 are ISSUES #56.
- Run on a machine with `data/raw/html_cache/`: dry-run
  `python scripts/build_merged_salaries.py`, confirm the moved rows match
  `merge_debug/expected_values.csv`, then run
  `python scripts/rebuild_training_data.py`, the suite, and
  `scripts/eval_stage3_signing.py`.
- Delete `outputs/experiments/merge_debug/` once the rebuild lands.
- Complete when the rebuilt frame is remeasured against the v8.17x baseline
  0.8422, 0.8595, and 0.8273 and the offsets are regenerated.

### Decide the ring-chasing discount

Four shapes measured on the pre-migration frame; the best is the gated
Stage-2 form in `scripts/eval_ringchase_gated.py` (hard age gate, P-weighted
multiplicative pull on the latent before the clip).

| shape | paired dSel t |
|---|---|
| two columns added to `FEATURE_COLS` | +1.22 |
| Stage-3 additive group offset | -3.03 |
| Stage-2 multiplicative, hard cell | -1.17 |
| Stage-2 multiplicative, gated + P-weighted | **+0.93** |

- Rerun the gated arm on the migrated frame. The age threshold is selected
  in-pool because 35 was reached by scanning and is not pre-registered.
- The blocker is not the shape: the profile scores discounters and
  top-dollar signers alike. Jimmy Butler 2023 (33, $218M banked, no ring)
  signed $45.2M and the pull deepened his error by $4.19M, while gains of the
  same size land on Conley and Horford. Report gain and damage by named row.
- LeBron James 2026 (41, two-year PHI minimum) is the largest forward miss:
  about $39.5M over-prediction on v8.17x. On v8.16x the 2026 origin read 0.885
  without him against 0.793 with him. Report his row and the 2026 origin with and without the pull. See
  `docs/briefs/2026-09-28-public-accuracy-views.RESULT.md`.
- Complete when it is adopted or rejected with paired metrics, a fixed subset
  definition, and per-row gain/damage reported.

### Improve the waiver features

Production includes `is_waived` and `mpg_x_waived`. Kemba Walker 2021 shows
that the binary signal cannot describe a value-dependent discount after a
buyout.

- Audit each positive `is_waived` event against information available before
  the signing date. Separate an ordinary waiver from a buyout re-signing.
- Remeasure `is_waived` and `mpg_x_waived` on the fixed v8.17x frame.
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

The v8.16x screen found that stronger settings shrink both bound-zone biases
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

- From the machine that holds the tags, run `git push origin --tags`. Tag any
  missing v8.x release commit with its headline metrics.
- Tag v8.17x on the merged commit with its headline metrics.
- Complete when `git ls-remote --tags origin` lists `pre-tidy-2026-09-28`,
  v8.16x, and v8.17x.

### Add P(max) to the Value Board

- Display max probability and conditional value, for example
  `85% max, $46.4M if maxed`.
- Complete when the board displays it without changing model scoring.

## Recurring

- Refresh the 2027 free-agent class with the chain in `CLAUDE.md`.
- Automate team-continuity refresh as maintenance. Treat it as a model feature
  only after a new evaluation brief.
