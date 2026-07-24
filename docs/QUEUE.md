# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-24, after the v7.10x landing.

## In flight

- **Censor-widening experiment** — worker out (brief
  `2026-07-24-censor-widening.md`, pinned 4156404). Win line: true-max zone
  ≥ $0.30M better, counterweight band bias growth ≤ $0.3M. If it wins it
  lands as the interim; the unified architecture then challenges the new
  incumbent.

## Next up (order matters)

1. **Unified architecture** — the signing-route probability function +
   per-route Stage-2 corrections (max/min/MLE snap branches from CBA
   numbers; continuous routes as shared f(x) + per-route fold-honest δ).
   P is an OUTPUT composition weight, never a feature (the rejected
   P(mechanism|x)-as-input, −0.0073, is a different thing). Branches adopt
   one at a time: max (push-then-clip) → min → MLE (needs a features-only
   separability AUC first; amounts table already curated) → continuous
   split + δ. Calibration judged on the predicted-band bias table, not the
   global slope. Brief waits only on the censor-widening result.
2. **Feature batch** (rides with the architecture work, same paired
   harness): (a) trend features (1-yr delta / 3-yr slope / peak-minus-
   current); (b) stats-as-of-signing for extensions; (c) Arm B prev_cap_pct
   semantics swap (+0.0020 over A, t=7, needs ship-form verification);
   (d) prior-year estimated value darko×mpg×$/win — new-column arm and
   rookie-exit in-slot arm; (e) rookie-award tier ordinal (ROY/All-Rookie
   1st/2nd/none — check award_score_cum's existing weighting first).
   Everything that fails the regression gates still feeds the route
   classifier.

## Parking lot

Wingspan (wingspan_minus_height), external archetypes, prehistory
expansion, playoff minutes share, early_gap>=2 mechanical ceiling rule
(ISSUES #6), Kanter→Freedom name alias (ISSUES #17). Docs agent: ISSUES
#10 and #15 (sigma mechanism; v7.9x/v7.10x records).

## Recently landed (context)

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
