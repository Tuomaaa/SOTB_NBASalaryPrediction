# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-25, after the censor-widening landing.

## In flight / ready to launch

- **Feature batch** — brief ready: `2026-07-25-feature-batch.md`. Five arms
  (trend / stats-as-of-signing / Arm B swap / prior-year est value /
  rookie-award tier), full challenger gates each, failed arms still feed the
  classifier.
- **Route mixture, phase 1** — brief ready: `2026-07-25-route-mixture.md`.
  The multiclass route classifier + the MAX branch (push-then-clip, fixed
  1.05 margin), win bar $0.50M on the true-max zone with the counterweight
  band and 25%+ predicted band as brakes. Adopting it starts the v8.0 line.
  Can run in parallel with the feature batch (machinery is feature-agnostic;
  re-run with batch winners at landing).
- Later phases on the same machinery: floor branch → MLE branch (conditional
  on the P(mle) separability AUC) → continuous split + per-route δ
  (Bird retention premium, possibly split extension/re-sign).

## Parking lot

Wingspan (wingspan_minus_height), external archetypes, prehistory
expansion, playoff minutes share, early_gap>=2 mechanical ceiling rule
(ISSUES #6), Kanter→Freedom name alias (ISSUES #17). Docs agent: ISSUES
#10 and #15 (sigma mechanism; v7.9x/v7.10x records).

## Recently landed (context)

- Censor-widening experiment — NO WINNER (e8e3c6a). The premise fell
  informatively: the counterweight band is itself underpredicted $4.6M, so
  a global dial has no interior optimum; ceiling for a calibrated global
  push is −$0.26M. Inert censor_c hook shipped; everything funnels to the
  per-row P.
- v7.10x — prev_cap_pct repair (paired +0.0135, t=7.6) + 2026 cap
  correction to the official 164,961,000; frame 944. Rookie-exit anchor
  arms all rejected (slot-average fails C2 despite best headline); the
  breakout-class bias (−3.16) is assigned to the max classifier.
- v7.9x — continuation filter v2 (dated spans; 342 demotions, is2019
  signal eliminated).
- MLE mass-point recon — 67 exact / 93 within 2%; snap oracle +0.0143;
  branch conditional on separability.
- AAV target switch — REJECTED (denominator artifact; cap-weighted mean
  share of a 5-yr max = 1.0029 of year-1).
- Sigma/gate retune — incumbent held; zone MAE is monotone in sigma (the
  clip makes overshoot free); never select sigma on zone MAE.
