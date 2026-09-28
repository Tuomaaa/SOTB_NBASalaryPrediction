# Work queue

Last updated 2026-08-31 after rejecting the Spotrac injury feature.

Keep only active work here. Each item needs an action and a completion check.
Use a dated brief for additional detail. Put defects in `ISSUES.md` and landed
work in `VERSION_HISTORY.md`.

## In progress

### Finish the Spotrac salary migration

- Resolve the 2019 COVID salary treatment, rebuild the frame, and compare it
  with HEAD on fixed rows. The current `split` branch recovered 61.4% of 57
  ambiguous rows against a documented 68% ceiling; taking the first Spotrac
  amount performed better.
- Complete when A1, A2, and B1 are remeasured against HEAD's 0.856, 0.881, and
  0.858; `SIGNING_OFFSETS_DEPLOYED`, `career_earnings.csv`, and
  `rings_thru_prev.csv` are regenerated from the final salary frame.

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
- Complete when it is adopted or rejected with paired metrics, a fixed subset
  definition, and per-row gain/damage reported.

### Improve the waiver features

Production includes `is_waived` and `mpg_x_waived`. Kemba Walker 2021 shows
that the binary signal cannot describe a value-dependent discount after a
buyout.

- Audit each positive `is_waived` event against information available before
  the signing date. Separate an ordinary waiver from a buyout re-signing.
- Remeasure `is_waived` and `mpg_x_waived` on the fixed v8.16x frame.
- Add `kf_market_value_x_waived = is_waived * kf_market_value` as a challenger.
  Build this interaction inside each CV fold after nested KF inference.
- Report paired A1, A2, B1, C1, and C2. Report named-row effects for Kemba
  Walker 2021 and Damian Lillard 2025.
- Complete when the audit is clean and each challenger passes the gates or has
  a recorded rejection.

### Grabit hyperparameter tuning — PAUSED

Paused by user decision on 2026-08-31. Keep the v8.16x incumbent
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

### Add P(max) to the Value Board

- Display max probability and conditional value, for example
  `85% max, $46.4M if maxed`.
- Complete when the board displays it without changing model scoring.

## Recurring

- Refresh the 2027 free-agent class with the chain in `CLAUDE.md`.
- Automate team-continuity refresh as maintenance. Treat it as a model feature
  only after a new evaluation brief.
