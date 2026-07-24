# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-25, after the censor-widening landing.

## In flight / ready to launch

- **Service years (ISSUES #20)** — brief dispatched:
  `2026-07-26-service-years.md`. **Run this BEFORE phase 3** (or re-pin
  phase 3 after it lands): it moves the max zone 68 → 70 and removes
  Reaves 2026 from the phase-3 collateral list, where he is the #2
  brake-killer at +$9.71M. Audit that scoped it: of the 53 rows landing
  exactly on a CBA tier (knife-edge — 10,000× the tolerance adds 4 rows),
  exactly 2 carry a ceiling above the tier they landed on (Reaves 2026,
  Butler 2019), both from the `age − 19` service fallback that covers 24%
  of rows. NOTE the design constraint written into the brief: the ceiling
  must stay computable BEFORE signing, so the exact-tier hit is a
  DETECTOR and the fix is a real debut-season source — never
  `ceiling = the tier the salary landed on`.

- **Route-mixture phase 3** — brief dispatched:
  `2026-07-26-route-mixture-p3.md`. Phase 2's "structurally unwinnable"
  verdict was an ISSUES #19 artifact: on corrected labels (v7.12x) the
  architect's re-run of the worker's own harness gives 90%-purity τ=0.64,
  36/68 touchable maxes, honest ceiling $0.85M, realized true-max
  **−$0.83M — the Win gate PASSES**. It now fails the brakes instead
  (25%+ band +$1.08M, counterweight +$0.36M vs +$0.30M), with only 4
  collateral rows and the damage concentrated in Aldridge 2019 (+$10.9M)
  and Reaves 2026 (+$9.7M, P 0.739 — excluded by any τ ≥ 0.80). Phase 3
  re-registers τ as "smallest τ with purity 1.000" (still probability-space,
  no zone metric) and runs {base, enriched} × {τ*, τ₉₀}. A passing cell is
  the v8.0 candidate.

- **Route-mixture phase 2** — brief dispatched:
  `2026-07-25-route-mixture-p2.md`. Enriched classifier (feature-batch
  columns as classifier-only inputs) → threshold by calibration purity
  (≥90% or stop) → gated push-then-clip, same win/brake battery. NOTE: the
  earlier "tier-max signings mislabeled" reading of the P>0.5 rows was
  OVERTURNED — a separate agent's row review (in progress) reads them as
  genuine classifier false positives; class definition unchanged, no
  tier-aware targets. The worker reports its touched-row collateral list
  for cross-checking against that review when it lands.
  **RECONCILED 2026-07-24 (v7.12x, cap-arithmetic verified):** the P>0.5 set
  is a MIX. (a) Kawhi 2019 / AD 2020 / Kemba 2019 / Kyrie 2019 / Mitchell 2025
  / Beal 2021 were genuine 30% maxes the is_max LABEL mis-tiered (award path
  granted 35% without a team-continuity check) — a real bug, NOW FIXED via
  designated_ineligible.csv; is_max 56→68, they read max, no longer "false
  positives." (b) Reaves 2026 / Anunoby 2024 ARE genuine false positives (not
  maxes) — the gate rejects these. Both readings were half-right. **Phase-2
  worker pinned pre-v7.12x must RE-RUN on the 68-max frame** (labels + zone
  n=68/MAE 4.44 moved; the mislabeled-max confusion is gone from the smear).
- **Route mixture, phase 2** — after the feature batch lands: (i) re-run the
  classifier on the batch winners (22 true maxes still under P=0.3, half
  Booker-class); (ii) precision-gated push — push only above a P threshold
  chosen from the CALIBRATION TABLE (probability-space rule, never tuned on
  zone MAE); pick the threshold on CALIBRATION PURITY, not just P height.
  The 44 smeared rows split three ways and MUST be hand-classified before
  phase 2 (2026-07-25 review, twice-corrected): (1) **genuine maxes the label
  mis-tiers** — Kawhi 2019 / AD 2020 / Kemba 2019 / Butler 2019 / Kyrie 2019
  signed real 30% maxes but `_compute_max_eligible` grants them a 35% ceiling
  off All-NBA without the own-team requirement, so is_max misses them
  (ISSUES #19, a real bug — fix it and the max zone grows, these stop being
  "false positives"). (2) classifier FALSE POSITIVES — good-but-not-max
  players the metrics oversold, **Reaves 2026 (signed his largest LEGAL deal,
  Early Bird, ~25% by coincidence, NOT a max)**, Anunoby 2024 — these are what
  the gate must reject, and they sit as high as P=0.74, so the [0.7,0.9) bin
  is only 72% pure. The safe threshold may be high enough the branch only
  touches ironclad maxes — consistent with the −$0.26M global ceiling. (3)
  fallen-star leak (Oladipo/Lillard/Drummond at P 0.2-0.3, $20-30M push
  damage) — killed by any gate, belongs to P(floor). Do NOT build a
  tier-aware push target that trusts P — it would push the Reaves class to a
  tier they never signed. Landing ISSUES #19 FIRST is the cleanest sequence:
  it removes class (1) from the confusion entirely. Then the floor
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

- v7.12x (61bdfcd) — designated-ceiling award-path fix (ISSUES #19). The
  award path granted the 35%/30% designated ceiling from All-NBA + experience
  without the CBA team-continuity requirement, so 12 genuine maxes were
  mislabeled non-max and their Stage-2 clip sat a tier high. Curated
  designated_ineligible.csv (mirror of early_supermax) reverts them to their
  real tier: is_max 56→68, Grabit zone MAE 6.18→4.44, A1 0.7849→0.7878 (same
  944 rows), confirmation 0.8090. Fixes the "smeared row" confusion at its
  source. Found via the route-mixture phase-1 review; the user caught two of
  my wrong reads (Kawhi did NOT take a pay cut; KAT did not trigger Rose)
  before it was right.

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
