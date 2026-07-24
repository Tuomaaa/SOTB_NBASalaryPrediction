# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-25, after the censor-widening landing.

## In flight / ready to launch

- **Feature batch** — brief ready: `2026-07-25-feature-batch.md`. Five arms
  (trend / stats-as-of-signing / Arm B swap / prior-year est value /
  rookie-award tier), full challenger gates each, failed arms still feed the
  classifier.
- **Route mixture, phase 2** — after the feature batch lands: (i) re-run the
  classifier on the batch winners (22 true maxes still under P=0.3, half
  Booker-class); (ii) precision-gated push — push only above a P threshold
  chosen from the CALIBRATION TABLE (probability-space rule, never tuned on
  zone MAE); the P≥0.7 region is ≥92% pinned and near-pure; (iii) **tier-aware
  branch target** (2026-07-25 row-level review of the 44 smeared rows): most
  P>0.5 "false positives" are REAL tier-max signings the is_max label misses —
  players signed AT the 25/30% tier while eligible for a higher one (Morant
  2023, KAT 2019, AD 2020, Kemba/Kawhi 2019, Reaves 2026). Push to the nearest
  tier at/above the prediction, not to max_eligible (Reaves: pred 38.3 → 25%
  tier 41.2 = his exact signing); consider widening the max CLASS to
  paid-at-any-tier. The fallen-star leak (Oladipo/Lillard/Drummond at P
  0.2-0.3 with $20-30M push damage) is killed by the precision gate and
  belongs to P(floor). Then the floor
  branch (P(floor) AUC 0.8233, pull-down mirror, same brake discipline).
  MLE branch HOLD: P(mle) AUC 0.7158 — team-cap membership invisible.
  A gating branch starts the v8.0 line and needs a separate predict.py
  wiring step (production ships plain XGBoost, no Stage 2 — worker-flagged
  integration caveat).

## Parking lot

Wingspan (wingspan_minus_height), external archetypes, prehistory
expansion, playoff minutes share, early_gap>=2 mechanical ceiling rule
(ISSUES #6), Kanter→Freedom name alias (ISSUES #17). Docs agent: ISSUES
#10 and #15 (sigma mechanism; v7.9x/v7.10x records).

## Recently landed (context)

- Route-mixture phase 1 (a040144) — classifier machinery landed, flag OFF,
  bit-identical when off. P(max) AUC 0.9650, median P on true maxes 0.52
  (0.36 pre-repair — the prev_cap_pct fix cashing in). Max branch did NOT
  gate: wins its zone −$1.59M (3× the bar) but raw-P continuous push smears
  moderate P onto ~49 non-max rows — brakes +$2.0/+$2.2M, B1 −0.017
  (origin-2025 big-dollar overshoots). Verdict: form problem, not
  classifier problem; see phase 2 above. No version number.
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
