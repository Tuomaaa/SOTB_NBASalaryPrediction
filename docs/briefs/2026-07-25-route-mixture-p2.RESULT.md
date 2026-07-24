# RESULT — enriched classifier + purity-gated max branch (phase 2)

**Brief**: `docs/briefs/2026-07-25-route-mixture-p2.md`
**Pin**: `master` @ af3222d (v7.11x line, 944-row frame, prev-season `prev_cap_pct`).
Worktree branch `worker/route-mixture-p2`.
**Verdict**: **the gated max branch DOES NOT gate — and this time it fails for the
opposite reason to phase 1.** Purity-gating at the pre-registered τ=0.90 removes
*all* collateral (0 non-max rows pushed; every brake and guardrail passes
trivially), but it can touch only 8 of 56 true maxes — and those 8 are precisely
the maxes the champion already prices correctly (mean champion error $0.12M). The
honest win ceiling is **$0.02M**, 25× below the $0.50M bar. The mechanism is a
structural anti-correlation, not a tuning miss: Spearman(P(max), |champion
error|) = **−0.681** across the 56 maxes. Recommend **do not adopt**; keep the
classifier as inert machinery (flag default off), and see §7 for why more
classifier capacity cannot rescue this architecture. The enrichment also
**failed to sharpen the classifier** (AUC 0.9653 → 0.9557), so it should not be
adopted into the router either.

---

## 1. What changed and why

Two changes, both classifier-side and both inert to the shipped champion. First,
`src/model/route_mixture.py` gained `attach_clf_features()`, which merges the
delivered feature batch (`data/processed/feature_batch_columns.csv`) onto the
frame as **classifier-only** inputs — 20 columns in three families (10 impact-
metric trend derivatives + coverage flag, 6 season-over-season sign deltas, and
`is_priced_early` / `est_value_prev` / `rookie_award_tier` /
`award_score_cum_clean`) fed with **native NaN** (the tested ship form for a tree
classifier). These columns join the classifier's feature list only; they never
enter the regression's `FEATURE_COLS`, and P(max) is applied to the Grabit latent
as an OUTPUT weight, so no leakage path into the target exists — the classifier
is a different model with a different target (route class, not `cap_pct`), and its
output is composed onto the regression's output, the opposite side of the ledger
from the ablation graveyard's `P(mechanism|x)` = −0.0073 input-side result.
Second, `make_maxbranch_fitter()` gained a `tau` gate and a `clf_features`
argument: rows with P(max) < τ are left at the champion prediction untouched.
The flag stays default **off**; with it off the fitter reproduces the champion
bit-for-bit (`fold_r2_selection` max|diff| vs the stored suite = **0.00e+00**).

The result is a clean negative. Phase 1 failed on the *brakes* — a continuous
raw-P push smeared moderate probability onto non-max rows and inflated their
bias. Purity-gating fixes that completely: at τ=0.90 the classifier's P≥τ set is
100% true-max (0 non-max), so the brakes, C2, ΔSel and B1 all pass. But the same
gate that removes the collateral also removes the *win*: the only maxes confident
enough to clear τ are the ones already sitting on their ceiling, because
classifier confidence and champion accuracy both key off the impact metrics. The
enrichment was supposed to sharpen the router; instead it lowered AUC while
marginally cleaning the extreme top — enough to drop the 90%-purity operating
point from τ=0.95 (base, 2 rows) to τ=0.90 (enriched, 8 rows), but those 8 extra-
covered rows carry essentially no error to win.

---

## 2. Part 1 — enriched classifier diagnostics (10-seed fold-honest OOF)

The enrichment columns are **classifier-only**; the base column below is the
phase-1 architecture (FEATURE_COLS only) re-run on the current v7.11x pin, so the
enriched-vs-base delta is same-frame, same-folds, same-seeds.

| classifier | P(max) AUC | median P on true max | non-max P>0.5 | P(floor) AUC | P(mle) AUC |
|---|---:|---:|---:|---:|---:|
| phase-1 published (v7.10x) | 0.9650 | 0.5217 | 13 | 0.8233 | 0.7158 |
| **base** (FEATURE_COLS, v7.11x pin) | 0.9653 | 0.5159 | 14 | 0.8253 | 0.7123 |
| **enriched** (+20 batch cols, 34 feat) | **0.9557** | **0.4169** | **13** | 0.8243 | 0.7019 |

**The enrichment did not sharpen the classifier — it slightly hurt it.** P(max)
AUC fell 0.9653 → 0.9557 and the median P on true maxes fell 0.5159 → 0.4169:
20 extra features on 56 positives is a capacity/noise trade the OOF sees through.
P(floor) is flat (0.8253 → 0.8243) and P(mle) drops (0.7123 → 0.7019). The base
classifier reproduces the phase-1 baselines almost exactly, confirming the pin
bump (v7.10x → v7.11x, prev-season `prev_cap_pct`) left the router unchanged.

**Enriched calibration (binned by predicted P):** well-calibrated in the middle,
and — the one thing the enrichment buys — a **100%-pure top bin**:

```
P(max) bin    n     meanP  emp_rate       (base [0.9,1.0): n=8 meanP 0.928 emp 0.875)
[0.0,0.1)   857    0.007    0.016
[0.1,0.3)    33    0.189    0.303
[0.3,0.5)    18    0.391    0.500
[0.5,0.7)    15    0.609    0.533
[0.7,0.9)    13    0.802    0.538
[0.9,1.0)     8    0.938    1.000   <- 8/8 true max
```

**Purity curve (cumulative: true-max rate among OOF rows with P ≥ τ).** This is
the object τ is chosen from, fixed before any zone metric:

```
 tau |  BASE n  nMax  purity |  ENR n  nMax  purity
0.50 |     42    28   0.667  |    36    23   0.639
0.55 |     34    25   0.735  |    33    20   0.606
0.60 |     32    24   0.750  |    29    18   0.621
0.65 |     31    23   0.742  |    25    16   0.640
0.70 |     26    19   0.731  |    21    15   0.714
0.75 |     22    17   0.773  |    19    13   0.684
0.80 |     16    13   0.812  |    15    11   0.733
0.85 |     13    11   0.846  |    11     9   0.818
0.90 |      8     7   0.875  |     8     8   1.000
0.95 |      2     2   1.000  |     2     2   1.000
```

Below τ≈0.85 the enriched curve is actually *less* pure than base (the lower AUC
showing through); its only advantage is the clean top bin, which is exactly what
the τ rule rewards.

---

## 3. Part 2 — τ from the purity curve (fixed before any zone metric)

Rule: **smallest τ with OOF purity ≥ 90%, τ ≤ 0.95.**

- **Enriched → τ = 0.90** (purity **1.000**, n≥τ = 8, all 8 true max, **0 non-max**).
- Base would need τ = 0.95 (purity 1.000, n = 2). The enriched classifier's sole
  contribution is moving the 90%-purity point down 0.95 → 0.90, tripling coverage
  2 → 8 maxes — reported for the record; it does not change the verdict (§4).

τ = 0.90 is used for the branch. This is a legitimate operating point (the brief's
"stop at Part 1" branch does **not** trigger — 90% purity *is* reached at
τ ≤ 0.95), so Part 3 runs.

---

## 4. Part 3 — the gated max branch (τ=0.90, fixed v7.11x rows)

Champion reproduction is exact (`fold_r2_selection` max|diff| vs stored suite =
**0.00e+00**), so every delta below is candidate-minus-champion on identical
(fold, seed) cells.

**Champion references (fixed rows):** true-max zone (n=56) MAE **$5.47M**, bias
−$5.47M; counterweight band ([0.70,0.90) of ceiling, non-max, n=44) bias
**−$4.11M**, MAE $6.56M; 25%+ predicted band (n=48) bias **+$0.79M**;
A1 0.7856, A2 0.8349, B1 0.8298.

**Honest win ceiling for τ=0.90** (computed before the run, per the brief):
8 of 56 true maxes clear τ; their summed champion error is $0.98M, so the maximum
*possible* true-max zone-MAE improvement is **$0.02M** — 25× under the $0.50M bar
before a single push is applied.

| arm | maxMAE | ΔMAE | cwBias | Δcw | 25band | Δ25 | ΔSel ± SE | t | A1 | A2 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| champion | 5.47 | — | −4.11 | — | +0.79 | — | — | — | 0.7856 | 0.8349 |
| **push_clip_gated** (ship) | 5.46 | **−0.01** | −4.11 | +0.01 | +0.80 | +0.01 | −0.00002 ± 0.00003 | **−0.76** | 0.7856 | 0.8349 |
| hard_gated | 5.46 | −0.01 | −4.11 | +0.01 | +0.80 | +0.01 | −0.00002 ± 0.00003 | −0.76 | 0.7856 | 0.8349 |

push_clip_gated per-fold ΔSel: `[0.0, 0.0, −0.00017, +0.00003, 0.0]`
(hard_gated identical — at τ=0.90 the push already lands the 8 rows on the ceiling,
so soft-push and hard-switch coincide).

The realized win is $0.01M (half the $0.02M ceiling; the soft push doesn't fully
pin). Everything else is a rounding-error no-op: A1/A2/B1 unchanged to 4 dp.

**C2 fixed-segment bias growth (push − champion, same rows):** every segment
Δ ≤ $0.01M — **no breach**.

```
Bird Rights  n=274  −2.49→−2.48 (Δ+0.00)    Minimum    n=242  +1.97→+1.97 (Δ+0.00)
Cap Space    n= 76  −0.15→−0.15 (Δ+0.00)    Non-Bird   n= 20  +2.11→+2.11 (Δ+0.00)
Early Bird   n= 46  −1.71→−1.71 (Δ+0.00)    Other      n= 18  +0.78→+0.78 (Δ+0.00)
MLE          n=123  +0.61→+0.61 (Δ+0.00)    Sign&Trade n= 13  −4.65→−4.65 (Δ+0.00)
Unknown      n=131  +0.37→+0.37 (Δ+0.00)
```

**B1 forward (rolling-origin 2024-26):** champion 0.8298, push_clip_gated 0.8298,
**drop −0.0000** (the push nudges 2025's max-zone MAE $2.82M→$2.69M and is
otherwise inert). This is the mirror image of phase 1, whose ungated push dropped
B1 by 0.0171.

```
origin  champ R2  push R2   champ maxzone MAE  push maxzone MAE
2024     0.8644    0.8644        $4.21M            $4.19M
2025     0.8193    0.8193        $2.82M            $2.69M
2026     0.7928    0.7928       $13.97M           $13.97M
```

---

## 5. Gate verdicts

| # | gate | threshold | push_clip_gated | verdict |
|---|------|-----------|-----------------|---------|
| 1 | **Win**: true-max zone MAE improvement | ≥ $0.50M | **$0.01M** (ceiling $0.02M) | ❌ **FAIL** (structural) |
| 2a | **Brake**: counterweight band signed-bias growth | ≤ +$0.30M | +$0.01M | ✅ PASS |
| 2b | **Brake**: 25%+ predicted-band bias growth | ≤ +$0.30M | +$0.01M | ✅ PASS |
| 3a | **Guard**: ΔSel t | > −2 | −0.76 | ✅ PASS |
| 3b | **Guard**: C2 fixed-segment \|bias\| growth | ≤ $0.30M | max Δ $0.01M | ✅ PASS |
| 3c | **Guard**: B1 forward drop | ≤ 0.003 | −0.0000 | ✅ PASS |

**Overall: does not gate — fails ONLY the Win gate, and fails it structurally.**
Purity-gating converted phase 1's five-way failure (both brakes + C2 + B1) into a
clean sweep of every brake and guardrail, at the cost of leaving nothing to win.

---

## 6. Collateral list (every non-max row above τ)

**Empty.** At τ=0.90 the enriched classifier's P≥τ set contains 0 non-max rows —
purity 1.000. This is the list the row-review agent's classification is to be
checked against: at the chosen operating point the max branch would push **no**
non-max rows, so the 13 non-max rows with P>0.5 (all with P<0.90) are irrelevant
to the branch either way, and any reclassification of those rows does not change
this result.

---

## 7. Anomalies & the mechanism (the real finding)

**Nothing scores "too good"; the escalation trigger does not fire.** The single
surprising number — the enrichment *lowering* AUC — is explained in §2 (capacity).
The interesting result is *why* a gate-passing brake battery still yields no win,
and it has a one-sentence mechanism:

> **The maxes the classifier is confident about are exactly the maxes the champion
> already prices correctly**, because both signals are the impact metrics.

Among the 56 true maxes, **Spearman(P(max), |champion error|) = −0.681**:

- **Touched** (P≥0.90, n=8: Luka '22, Mitchell '21, Zion '23, Franz Wagner '25,
  SGA '22, Tatum '25, LaMelo '24, Embiid '23): mean champion |error| **$0.12M** —
  all but LaMelo (−$0.57M) and Embiid (−$0.42M) are already dead-on. These look
  unambiguously like maxes on impact metrics → high Grabit latent → clipped to
  ceiling → nothing left to push.
- **Untouched** (P<0.90, n=48): mean champion |error| **$6.36M**, holding
  **$305.3M of the $306.3M ($99.7%) of total zone error**. These are the
  reputation-/extension-priced maxes the impact metrics discount and therefore
  the classifier is unsure of: Trae Young '26 (−$32.5M, P 0.03), JJJ '22/'26
  (−$11.3/−$28.5M, P 0.004/0.002), Myles Turner '22 (−$21.6M, P 0.008), MPJ '22
  (−$16.9M, P 0.02), Booker '19 (−$4.6M, P 0.07), Klay '19 (−$9.6M, P 0.20). Every
  large champion error sits far below τ **by construction**.

A purity-gated push is therefore structurally incapable of winning the max zone:
purity and winnable-error are the same axis with opposite sign.

**Capacity is not the lever (prune sensitivity).** Dropping the 10 sparse trend
derivatives (24 classifier features) recovers discrimination — AUC 0.9595, median
0.5203 — and lets τ fall to 0.85 (12 maxes, purity 0.917). But the newly-reachable
error is **83% one row**: John Wall 2019 (champion error −$14.19M), a post-injury
35% supermax the market arguably overpaid and the model *should* discount. Strip
Wall and the pruned honest ceiling collapses back toward $0.05M; keep him and it
is ~$0.30M — still under the $0.50M bar — while 1 non-max row leaks back above τ,
re-opening the brakes. No reachable classifier makes this branch gate: the win
requires touching maxes that don't look like maxes, and any classifier that
flags them also flags non-maxes.

**A note on the biggest "errors" (ties to ISSUES #19).** Several of the largest
untouched-max champion errors are label/tier artifacts, not market signal: John
Wall 2019, Bradley Beal 2022 (−$9.9M), KAT 2024 (−$6.5M) and the mis-tiered rows
ISSUES #19 documents. Chasing them with a push would "improve" zone MAE by fitting
contracts the market itself mispriced. This reinforces that the max zone's
residual is not a modelling miss the router can close.

---

## 8. Phase-3 recommendation — floor branch: **NO-GO in this form**

P(floor) enriched AUC is **0.8243** (base 0.8253, phase-1 0.8233) — the enrichment
does not help the floor router any more than the max router. More importantly, a
purity-gated floor branch is a *pull-down mirror* of this max branch and inherits
the identical structural obstacle: the floor is **already left-censored in Stage 1**
(v7.8x, k=2.0), so the rows a floor classifier is confident about are the ones the
left-censor already pins — the same touch-vs-win disjunction, at the other bound.
Recommend **do not build the floor branch**. If a future agent tries it anyway,
gate it on the floor zone with the same brake discipline and compute the honest
floor-zone win ceiling (champion error on high-P(floor) rows) *first* — expect it
to be small for the same reason. The route-mixture line's remaining value is
diagnostic (the classifier), not a structural branch.

The broader lesson for the architect: **the max/floor branches cannot be rescued
by a better router.** Winning either censored zone requires a signal *orthogonal*
to the impact metrics that identifies reputation-priced deals (team-continuity,
signing-date/tier structure — the ISSUES #6/#19 data gap). The feature batch is
not that signal; it lowered the router's AUC and left the touchable set unchanged
where it mattered.

---

## 9. Files touched & where the branch lives

Branch: **`worker/route-mixture-p2`** (worktree `../BBall-worker-route-mixture-p2`,
pinned to `master` @ af3222d).

- `src/model/route_mixture.py` (modified) — `attach_clf_features()` +
  `CLF_EXTRA_COLS`; `make_maxbranch_fitter()` gains `tau` gate and `clf_features`.
  Champion path unchanged (flag default off, bit-identical).
- `scripts/eval_route_mixture_p2.py` (new) — the evidence harness reproducing
  every number here: `OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p2.py`.

Artifacts (gitignored `outputs/models/`, for the architect's review):
- `route_mixture_p2_eval.json` — all Part 1/2/3 metrics.
- `route_mixture_p2_oof.csv` — per-row base/enriched P(max), P(floor), route
  label, champion + gated OOF.

**Not touched**: `train.py`, `evaluate_suite.py`, the target/filter chain, sigma/
gates/censor population, `FEATURE_COLS`, and the docs-agent lane. Frame unchanged
at 944 rows; the champion reproduces the stored reference exactly.

---

## 10. Proposed commit message (architect edits and lands)

```
Route-mixture phase 2: enriched classifier + purity-gated max branch (does not gate)

Add the feature batch as CLASSIFIER-ONLY inputs (attach_clf_features, 20 cols,
native NaN; never enter FEATURE_COLS) and a tau gate on the max branch, both
behind the flag default OFF — champion is bit-identical (fold_r2_sel max|diff|
0.00e+00).

The enrichment does NOT sharpen the router: P(max) AUC 0.9653 -> 0.9557, median
on true maxes 0.5159 -> 0.4169 (20 features, 56 positives). Its only effect is a
100%-pure top bin, moving the 90%-purity operating point from tau=0.95 (2 maxes)
to tau=0.90 (8 maxes).

Gated max branch at the pre-registered tau=0.90 does NOT gate and is NOT enabled.
Purity-gating removes ALL phase-1 collateral (0 non-max pushed; both brakes, C2,
dSel t=-0.76, and B1 drop -0.0000 all pass) but the honest win ceiling is $0.02M
vs the $0.50M bar: the 8 touchable maxes have mean champion error $0.12M while the
48 untouched maxes hold 99.7% of the zone error. Spearman(P, |champ err|) = -0.681
-- classifier confidence and champion accuracy are the same signal (impact
metrics), so purity and winnable-error are anti-correlated. Prune sensitivity
(tau=0.85, 12 maxes) confirms it is structural, not capacity: the reachable error
is 83% John Wall 2019, a mispriced supermax. Floor branch (phase 3): NO-GO, same
structural obstacle at the left bound. Analysis-only; no version number.
```

(No version number claimed — the brief reserves the v8.0 line for a gating branch;
it did not gate.)

---

## 11. ISSUES.md additions

No new defect. Appended a one-line corroboration to **ISSUES #19** (the
`_compute_max_eligible` mis-tiering): this phase-2 analysis independently
quantifies its cost — the mis-tiered/artifact maxes (John Wall 2019, and the
#19 rows) are among the largest champion errors in the max zone and are exactly
the rows a purity-gated branch cannot reach, so #19 is a genuine ceiling on any
future max-zone intervention, not just a label nit.
