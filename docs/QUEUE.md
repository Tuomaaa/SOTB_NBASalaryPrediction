# Work queue (durable copy — the in-app task board resets on restart)

Maintained by the architect session. One line per item; briefs under
`docs/briefs/`, decisions in the RESULT files and tags.

Last updated: 2026-07-26, after the five-worker landing (v7.13x).

## In flight / ready to launch

Five dispatched 2026-07-28, file ownership exclusive so they run in
parallel:

- **Wire stage 3** (`2026-07-28-wire-stage3.md`) — adopt push + extension
  clip into the pipeline. Decided with the user: push + clip, not clip alone
  (clip alone is cleaner on t=1.48 and the band brake but leaves the max
  zone's $1.12M on the table, which is the problem the line exists for).
  Measured A1 0.7865 → 0.7988, A2 0.8341 → 0.8517, B1 0.8342 → 0.8355,
  2026 origin 0.7964 → 0.7989. Adopted at t=1.11 on the same
  legal-bound-not-fitted-parameter grounds as v7.4x/v7.9x/v7.13x. Owns
  extension_cap / evaluate_suite / the Stage-2-3 composition / export_web /
  predict.
- **Ceiling consistency** (`2026-07-28-ceiling-consistency.md`) — ISSUES #27
  (Smart 2022 still reads a 35% tier ceiling in `_compute_max_eligible`; the
  #23 fix only reached `extension_cap`) and #26 (the `prev_cap_pct` FEATURE
  still carries pre-correction values while the ceilings use corrected ones).
  Moves the champion. Owns `_compute_max_eligible`, the curated ineligible
  list, and the training table.
- **Docs, pass 1** (`2026-07-28-docs-v79-v713.md`) — VERSION_HISTORY stops
  at v7.8x and five landed versions are frozen and cannot move, so they get
  written now. Owns `VERSION_HISTORY.md` alone; explicitly forbidden from
  writing v8.0x, touching METHODOLOGY, or describing the three-stage
  architecture — that is pass 2, after the wiring lands. Told to name
  problems rather than cite issue numbers, since the numbering is being
  repaired in parallel.
- **Offseason-injury probe** (`2026-07-28-offseason-injury-probe.md`) —
  reconnaissance only. The floor branch's +0.0396 needs a fact dated between
  the priced season and the signing, and the two injury files we hold cannot
  supply it: they are game-day availability logs (Jun-Sep records are 1.88%
  and 0.18% of rows) with holes at Oct-2020→Oct-2021 — which covers Oladipo
  2021, the largest floor error — and after June 2024. The probe establishes,
  on nine named rows, whether the event is recorded anywhere public, **counts
  the ones with no injury at all** (Harrell may just have hit a cold market),
  and computes the subset oracle before anyone funds a scrape.
- **ISSUES cleanup** (`2026-07-28-issues-cleanup.md`) — two `#20`s, two
  `#25`s, four fixed-but-undeleted entries, and #21's title now says the
  opposite of the truth. Owns `ISSUES.md` alone; the other two put their
  entries in their RESULTs and the architect slots them in.

**2026's forward number is two contracts.** Trae Young (−$32.1M) and JJJ
(−$25.0M) carry 50.5% of the 2026 origin's squared error; excluding them
takes it from 0.7964 to 0.8792, the best of the three origins. Both are
widely-panned deals the model correctly declines to endorse, so that part of
the residual should not be chased — the evaluation metric cannot tell "the
model is wrong" from "the market was wrong".

- **Told-clip + data fix** — brief dispatched:
  `2026-07-27-told-clip-and-data-fix.md`. Fixes ISSUES #23 (Smart 2022 gets a
  designated-veteran exemption from an award that POSTDATES his signing — he
  alone carries +$22.22M of the arm's error) and #22 (three over-cap rows,
  all defects in our salary table), then re-measures the told-route Stage-3
  clip at the pre-registered τ = 0.52.
- **Floor branch** — brief dispatched: `2026-07-27-floor-branch.md`. The
  largest unclaimed headroom in the project: oracle **+0.0396** (A1 0.7865 →
  0.8261) because the Stage-2 clip only pushes UP, so 78 of 240 floor rows
  over-priced by >$2M are untouched. Fat-tailed — the worst 12 rows carry 31%
  of the zone error and every one is a fallen star on a minimum (Oladipo
  2021 +$24.9M, Oubre 2023 +$18.1M, Harrell 2022 +$17.2M). Phase 2's NO-GO
  was an argument, not a measurement, and it is the same argument the max
  side disproved. **No legal brake exists on this side** — no CBA rule raises
  a player's floor by route — so the threshold and the selection rule are the
  only guards.

### Convention, 2026-07-26, refined 2026-07-27 (user's decisions)

The Stage-3 told-route number is reported as the model's accuracy in the SAME
column as the ex-ante numbers — the route is information the champion also has
and merely ignores, so using it is a covariate, not leakage. **v7.1x–v7.13x
were computed under the old "ignore the route" convention**; every RESULT and
version entry from here must say so, so the two are not compared naively.

**The refinement, and its test.** The convention applies PER ROUTE, and the
test is: *after being told the route, does the salary still require a
non-trivial computation?*

- **Extension — YES, counts.** Told "he extended", you still have to compute
  1.40 × prior pay (or the average-salary alternative) to land on Brunson's
  $34.94M. The bound is a real function of data.
- **Floor — NO, does not count.** Told "he signed at the floor", you are
  within **$0.13M** of his pay already (the floor-branch RESULT measures
  `floor_pct` against observed pay: MAE $0.130M). Being told the route IS
  being told the answer, so the told arm's +0.046 measures recitation, not
  prediction. It also has no deployed use: for an unsigned free agent
  "will he take the minimum" is precisely the quantity being predicted,
  whereas "he is not extending" is true by definition of being a free agent.

- **Extension route (ISSUES #21)** — brief dispatched:
  `2026-07-26-extension-route.md`. Implements the veteran-extension raise cap
  (120%/140%, greater of prior pay and the league average salary) with the
  rookie-scale/veteran split done on a proper instrument, gated on
  "rows paid above their own cap ≈ 0" — the check that caught the architect's
  two failed attempts. Then the user's full six-route mixture
  (max/floor/mle/bird/capspace/extension) with the centering discipline the
  δ failure taught, then a phase-3 re-run. Adjudicated in the brief: these
  rows are NOT right-censored — the cap binds only conditional on choosing to
  extend, the same "choice, not constraint" that killed censoring good
  players on minimums; Brunson 2025 took a widely-reported discount.

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
2. **Team-continuity signal — DEMOTED 2026-07-28** to a maintenance item,
   not an accuracy one (user's call, and the arithmetic agrees). Its three
   claimed payoffs do not survive costing: retiring the two curated lists is
   a maintenance win because those 14 rows are already *correct* by hand;
   δ_bird's entire ceiling under perfect route knowledge is ~+0.008 and its
   ex-ante form has failed every time; and better P(route) is the thing three
   separate experiments have shown does not convert. Worth doing so nobody
   hand-adds a row each summer — not worth doing for the metric.
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

- **Route-mixture max branch** — **CLOSURE PROVISIONAL, pending ISSUES #21.**
  Filed 2026-07-26: veteran-extension raise caps (120% pre-2023 CBA, 140%
  after) are not implemented, so 21 rows sit at a legal bound we cannot see
  and 15 carry ceilings up to $29.7M too high. Three of the five collateral
  rows that fail the brakes — Aldridge 2019 (1.16× prior), Smart 2022
  (1.26×), Brunson 2025 (**1.400× exactly**) — are extension-capped maxima,
  not classifier errors, and they carry 82% ($22.7M of $27.8M) of the push
  damage. The counterweight band the brake is built on may itself be
  populated by these rows, in which case its $4.87M "underprediction" is a
  data bug rather than model error. **Re-run the phase-3 harness after #21
  lands before treating this line as closed.** This would be the THIRD time
  a ceiling bug masqueraded as a modelling limit (#19 invalidated phase 2,
  service years invalidated phase 3's brakes).

- Closure rationale as it stood before #21 was found — closed after three phases. **The phase-3
  RESULT's numbers are stale**: that worker was pinned before the
  service-years landing, exactly the trap that invalidated phase 2. The
  architect re-ran its harness unmodified on v7.13x (2026-07-26); use these
  numbers, not the RESULT's.

  What the re-run changed: the counterweight brake now PASSES at the
  90%-purity points (0.62 → 0.19 base, 0.36 → 0.13 enriched) because Reaves
  and Butler left that band on becoming maxes, and the win survives the
  confirmation split (enriched τ90 wins $0.88M on decidable rows against the
  $0.50M bar) — so phase 3's headline argument, that the win was an artifact
  of the locked rows, no longer holds.

  What did not change — the closure: the two operating points fail for
  opposite and now fully understood reasons.
  · **Zero collateral (enriched τ=0.91, 26 maxes)**: every guardrail clean,
    pooled R² actually +0.0026, band MAE down — but the zone win is $0.36M
    against the $0.50M bar and ΔSel t = −0.29. Too few rows to win.
  · **90% purity (enriched τ=0.50, 48 maxes + 5 collateral)**: wins big, and
    the brake it "fails" is mis-specified — 70% of that band's +$1.26M bias
    growth is 37 at-ceiling maxes being CORRECTED (Beal 2022 −$9.05M → 0.00,
    Haliburton 2024 −$5.90M → 0.00), which a signed-bias instrument cannot
    tell apart from over-pushing. But the config dies anyway on the primary
    metric: pooled R² 0.7865 → 0.7842, ΔSel t = −1.73, because squared error
    punishes the few large overshoots (Aldridge 2019 alone +$10.41M).

  So the real constraint is not "purity and winnable error are the same
  axis" (phase 2/3's framing) but: **too few reachable rows to clear the bar,
  and reaching more always drags in a few catastrophic overshoots.**
  **SUPERSEDED 2026-07-28.** This closure did not survive: the user asked
  why the branch was being closed, and the two gates it failed turned out
  to be (a) a brake the architect had already recorded as mis-specified and
  (b) a forward drop carried by three extension rows the Stage-3 clip now
  catches. Push + clip is being wired in (see In flight). Any earlier line
  in this file calling team continuity 'the only route to reopen the max
  push' is void — the push was never killed.
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
- **Waiver interaction** — `prev_cap_pct × is_waived` (+0.00334, t=1.43) and
  `mpg × is_waived` (+0.00539, t=1.96) both fail gate 1 (need t>2). Oracle
  ceiling on the 11 top-band waived rows is +0.00546 at t=1.09 — **permanently
  closed**: no feature can pass the gate on this segment. Three fabricated
  salary rows found (Beal 2025, Noah 2020, Batum 2020 — ISSUES #36).
- **Two features (role mismatch, size for position)** — all four arms fail on
  both the original pin (v8.0x) and current master (v8.2x): role_mismatch
  +0.00065 t=+0.72, size_for_position +0.00044 t=+0.53. The interaction does
  not beat the already-rejected main effect.

## Standing decisions (do not re-propose without new evidence)

- **How a route earns a place on the EX-ANTE side of the mixture.** Decided
  2026-07-26, after the user challenged a wrong justification of mine (I had
  claimed the extension cap is ex-ante computable while the Bird premium is
  not — false: δ_bird is a population constant, equally computable; what is
  ex post is only *which route the player took*, and that is true of both).
  The real criterion is the product of two quantities:
  **(value gap from f(x)) × (unreliability of P)**. Measured so far:
  max +$4.37M at AUC 0.982 (earns its place), floor −$2.07M at 0.826,
  **bird +$1.61M at 0.811 — the worst combination, and the one that actually
  blew up** (every row got P×$1.61M, the surface lifted, calibration 0.985 →
  0.977, three gates failed), mle a hard snap at 0.710 (told-side only),
  capspace +$0.10M at 0.678 (δ≈0 — it does not deserve to be a separate route
  at all). Extension is UNMEASURED and gets no presumption either way; its
  operation is a truncation rather than an additive premium, so it cannot
  drift the whole surface, but it can systematically under-predict young
  risers who reach free agency instead — the mirror failure. Assign it from
  its measured headroom and P, not from an argument.
  **Expected landing shape**: ex ante = P(max)·V_max + P(floor)·V_floor +
  P(rest)·f(x); told-parameter side = mle snap, bird constant, extension
  truncation — three DIFFERENT operations, each defined and gated separately.
  When the extension-route bundle lands, judge each leg by this criterion, not
  merely by whether the six-route form as a whole gated.

- **An empirical regularity does not get to be a Stage-2 bound.** Decided
  2026-07-29, after the architect proposed one and the user rejected it. The
  proposal: waived players never sign above the mid-level (0 of 122 rows exceed
  15% of cap; the observed max is 12.5%), so clip them there. An in-sample clip
  on the champion's own OOF column reads **+0.0078** at the 15% line and
  +0.0111 at 8%. Rejected anyway. Stage 2's contract is *deterministic CBA
  bounds on the player himself* — the max tier and the league minimum — and it
  earns its exemption from the gates precisely because those are written down
  and cannot be wrong. A waiver ceiling is none of that: no CBA provision caps
  a waived player's next contract (the nearest real rule, that a team over the
  first apron cannot sign a player whose pre-waiver salary exceeded the
  non-taxpayer MLE, restricts the buyer list and not the amount); it is a
  one-way ratchet that would permanently forbid pricing a bought-out star
  correctly; and the 15% line is fitted to the sample maximum of 122 rows,
  the least stable statistic available. **The +0.0078 is not a bound's payoff —
  it is a measurement of Stage 1 failing to learn something it has the
  information to learn.** Clipping hides that failure; the correct response is
  the feature route (`2026-07-29-waiver-interaction.md`). General form of the
  rule: if it is not in the CBA, it belongs in Stage 1 and runs the gates.
- **The mechanism behind the waiver discount is the fact, not the money still
  owed.** Same date, same conversation. The architect proposed `dead_money_pct`
  (a bought-out player collects the old contract regardless, so the new deal is
  incremental income and he optimises for role over dollars); the user said it
  is being waived itself. The data sides with the user **on the sign**: the
  dead-money story predicts the landing spot FALLS as more money is already
  banked, and the fitted waived line has slope **+0.140** — positive, just
  heavily flattened from the non-waived +0.750. What a waiver does is erase
  ~81% of the price history, which is a public negative signal from the party
  with the most information, not a budget effect.

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
