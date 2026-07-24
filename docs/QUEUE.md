# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-25, after the censor-widening landing.

## In flight / ready to launch

- **Route-mixture phase 2 brief** — next to write. Prerequisites all in:
  feature-batch columns delivered, the 44 smeared rows' three-way split
  documented below (hand-classification against tx_text "maximum" keywords
  is the likely instrument), threshold by calibration purity.
- **Route mixture, phase 2** — after the feature batch lands: (i) re-run the
  classifier on the batch winners (22 true maxes still under P=0.3, half
  Booker-class); (ii) precision-gated push — push only above a P threshold
  chosen from the CALIBRATION TABLE (probability-space rule, never tuned on
  zone MAE); pick the threshold on CALIBRATION PURITY, not just P height.
  The 44 smeared rows split three ways and MUST be hand-classified before
  phase 2 (2026-07-25 review, corrected): (1) genuine tier-max signings the
  is_max label misses — signed AT a tier they qualified for, KAT 2019 / AD
  2020 / Kawhi 2019; widening the max CLASS to paid-at-any-tier is the fix
  for these. (2) classifier FALSE POSITIVES — good-but-not-max players the
  metrics oversold, e.g. **Reaves 2026 (signed his largest legal deal, ~25%
  by coincidence, NOT a max)** — these are what the gate must reject, and
  they sit as high as P=0.74, so the [0.7,0.9) bin is only 72% pure and even
  [0.9,1.0) is 78%. The safe threshold may be high enough that the branch
  only touches ironclad maxes — consistent with the −$0.26M global ceiling
  meaning little headroom remains. (3) fallen-star leak (Oladipo/Lillard/
  Drummond at P 0.2-0.3, $20-30M push damage) — killed by any gate, belongs
  to P(floor). Do NOT build a tier-aware push target that trusts P — it
  would push the Reaves class to a tier they never signed. Then the floor
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

- v7.11x — feature batch: only Arm C (prev_cap_pct = previous-season pay)
  adopted, at the gate boundary (+0.00178, fold-paired t=1.96; magnitude
  pre-registered, all instruments improve, semantics simplification).
  Four arms rejected: trend / est-value / rookie-award flat (existing
  features already carry the signal), stats-as-of-signing harmful (the
  market prices expected growth, so current-season stats beat signing-date
  stats for extensions; MPJ is the rare inversion). All columns delivered
  in data/processed/feature_batch_columns.csv for the phase-2 classifier.
  Protocol correction: the prev-cap RESULT's "t=7.0" was SEED-paired
  (canonical fold-paired t≈2); pairing unit is the fold, always. New
  ISSUES entry: awards_full name-join drops footnote-marked stars.
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
