# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-26, after the five-worker landing (v7.13x).

## In flight / ready to launch

Nothing. Five workers landed together on 2026-07-26 (v7.13x); the two
architecture lines below are decided, and what remains is listed under
"Open work".

## Open work, in priority order

1. **Ex-post δ as a Stage-2 mode** (the one adoptable thing the δ analysis
   found). δ_bird = +$1.61M is real and fold-stable; ex-post it scores
   ΔSel +0.00769 (t=2.37) and closes Bird bias −$2.44M → −$1.55M. It must
   NOT enter the headline CV — the headline is ex-ante, and using the
   realized route there would be scoring with outcome information. Its
   legitimate home is the told-parameter mode: Contract Surplus on a signed
   deal, and the Value Board's "given he re-signed with Bird rights" view.
   Needs a brief that defines the two modes in code and reports them
   separately, plus CONTEXT.md vocabulary for the split.
2. **Team-continuity signal** (ISSUES #6 maintenance debt). A signing-date
   team-match would replace BOTH curated lists (`early_supermax.csv`,
   `designated_ineligible.csv`), each of which needs a hand-added row every
   summer, and is also the missing feature behind δ_bird (the model has no
   team-history input at all — ISSUES #5). One signal, three payoffs.
3. **Product display** — P(max) on the Value Board ("85% max, $46.4M if
   maxed"). The classifier is strong enough now (AUC 0.9823, median P on a
   true max 0.81) and this needs no gate because it does not enter scoring.
4. Parking lot / low priority: ISSUES #4 (10% unlabeled mechanisms), #5
   (team+position missing, cosmetic), #12 (coverage-skew memo), #17's
   remaining two-way players, wingspan, archetypes, prehistory expansion,
   playoff minutes share.
5. **Seasonal**: the 2027 FA class refresh when it arrives (the four-command
   chain in CLAUDE.md).

## Closed architecture lines (do not reopen without new data)

- **Route-mixture max branch** — closed after three phases. Final state on
  corrected labels: the classifier is excellent (P(max) AUC 0.9823) but no
  operating point both wins and stays clean. Zero-collateral thresholds have
  honest ceilings of $0.15M/$0.43M — below the $0.50M bar before any arm
  runs — while 90%-purity thresholds clear the win and break both brakes.
  Spearman(P, |champion error|) = −0.761: the maxes a classifier can be
  confident about are the ones the champion already prices, because both key
  off the impact metrics. Reopening requires a signal ORTHOGONAL to the
  impact metrics (team continuity, reputation), not a better router.
- **Floor branch** — NO-GO by the same structure at the other bound (the
  floor is already left-censored in Stage 1, so high-P(floor) rows are the
  ones the censor already pins).
- **MLE branch** — HOLD. P(mle) AUC 0.71: whether a player is offered an
  exception depends on the signing team's cap position, which no player
  feature can see.
- **Ex-ante per-route δ** — fails gates 1, 4 and C1. Because mean P(bird) is
  0.29 on every row, P × δ_bird lifts the whole surface (MLE +$0.35M,
  Non-Bird +$0.29M, slope 0.985 → 0.977). The loss is in P(route), not in δ.
- **Extension vs re-sign inside Bird** — does not separate (n=31, diff
  −$1.40M against SE $1.12M). One δ for Bird.

## Standing decisions (do not re-propose without new evidence)

- **The Stage-2 clip never reads the observed salary — in either direction.**
  Clipping DOWN at actual pay is straight target leakage: over-prediction
  becomes impossible, so every bargain vanishes from the Value Board by
  construction and a model that predicts $40M for everyone scores well.
  Clipping UP to actual pay when the ceiling came out too low is logically
  defensible (pay ≤ legal max, so the salary certifies a lower bound on the
  true ceiling) but is still rejected, for two reasons: (a) that contradiction
  is currently our most sensitive data-error detector — it is how ISSUES #19's
  12 mis-tiered rows were found — and auto-repairing it silences the alarm,
  the same failure shape as ISSUES #16 (stale cap + stale salaries agreeing
  with each other); (b) `predict.py` has no salary for an unsigned free agent,
  so the guard cannot run at inference and CV would drift optimistic relative
  to deployment. **Correct form: assert loudly, fix the input.** Empirically
  the situation is already zero — the only rows above their ceiling are 3
  float-dust cases at ratio 1.0000 (Adebayo/Tatum 2021 at 0.250000, Giannis at
  0.350001), so any over-ceiling check should use the same 1e-4 tolerance
  `_filter_mislabeled_year1` uses. Decided 2026-07-26.

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
