# RESULT: per-route δ for the continuous routes

**Brief**: `docs/briefs/2026-07-26-route-delta.md`
**Pin**: `master` (df8c268)
**Branch**: `worker/route-delta`

## What changed and why

The continuous signing-route class — everything that is not max, floor, or
MLE — was split into **bird** (Bird Rights + Early Bird + Non-Bird, n=264),
**capspace** (Cap Space, n=69), and **other_cont** (Sign & Trade, Unknown,
Other, Minimum leftovers, Rookie Scale, n=210). A 6-class route classifier
was trained on the extended set, and a per-route intercept correction
δ_k = mean(OOF residual for route k) was estimated fold-honestly.

The δ idea is sound: the ex-post oracle (f(x) + δ_route using the **true**
route) passes gate 1 at t = 2.37 and closes Bird bias from −$2.44M to
−$1.55M. But the ex-ante form (Σ P(k) × (f(x) + δ_k)) fails two gates —
the classifier is not accurate enough for the δ to survive composition with
P(route). The loss is localized to P(route), not to δ.

## Numbers table

### Classifier diagnostics (Part 1)

| Class | n | OvR AUC | P̄ | Actual | Ratio |
|---|---|---|---|---|---|
| bird | 264 | **0.8113** | 0.2924 | 0.2797 | 1.05 |
| capspace | 69 | 0.6782 | 0.0695 | 0.0731 | 0.95 |
| other_cont | 210 | 0.6826 | 0.2265 | 0.2225 | 1.02 |
| max | 68 | 0.9755 | 0.0614 | 0.0720 | 0.85 |
| mle | 93 | 0.7095 | 0.0921 | 0.0985 | 0.93 |
| floor | 240 | 0.8294 | 0.2582 | 0.2542 | 1.02 |

P(bird) AUC = 0.8113. Among TRUE bird rows: mean P(bird) = 0.507, median
= 0.509. Among non-bird continuous rows: mean P(bird) = 0.223. This is
substantially stronger than P(mle) (AUC 0.7095) but the continuous routes
lack the hard CBA cliffs that make the structural routes identifiable.

### δ table (Part 2)

| Route | n | δ (cap_pct) | SE | $M | Per-fold δ |
|---|---|---|---|---|---|
| bird | 264 | **+0.01195** | 0.00244 | **+$1.61M** | +0.014, +0.010, +0.014, +0.010, +0.012 |
| capspace | 69 | +0.00083 | 0.00498 | +$0.10M | −0.000, −0.004, +0.006, +0.002, +0.001 |
| other_cont | 210 | +0.00045 | 0.00226 | +$0.06M | +0.001, +0.000, −0.000, +0.000, +0.001 |
| max | 68 | +0.03336 | 0.00564 | +$4.37M | +0.039, +0.027, +0.031, +0.036, +0.035 |
| mle | 93 | −0.00439 | 0.00319 | −$0.57M | −0.004, −0.005, −0.007, −0.003, −0.004 |
| floor | 240 | −0.01612 | 0.00178 | −$2.07M | −0.015, −0.017, −0.016, −0.016, −0.015 |

δ_bird is the only continuous-route δ that is both large and stable across
folds. The capspace and other_cont δs are indistinguishable from zero.

#### Extension vs re-sign inside Bird

| Subgroup | n | Mean resid | SE | $M |
|---|---|---|---|---|
| Extensions | 31 | +0.00273 | 0.01080 | +$0.39M |
| Re-signings | 165 | +0.01318 | 0.00285 | +$1.78M |
| Unknown | 68 | +0.01316 | 0.00422 | +$1.71M |

ext − resign diff = −0.01045 ± 0.01117 (−$1.40M). |diff| / max(SE) = 0.97
— the difference does NOT exceed its SEs. **One δ for Bird.**

The direction is suggestive (extensions show less premium than re-signings),
but n=31 with SE > |mean| is not evidence. The unknown group (n=68) tracks
re-signings closely.

### Evaluation battery (Part 3)

| Metric | Champion | Ex-ante | Ex-post |
|---|---|---|---|
| A1 CV R² | 0.7878 | 0.7900 | **0.7926** |
| A2 CV R² (2024-26) | 0.8335 | 0.8356 | 0.8343 |
| B1 Forward R² | 0.8303 | 0.8308 | 0.8287 |
| ΔSel (paired) | — | **+0.00283** | **+0.00769** |
| SE | — | 0.00316 | 0.00325 |
| t | — | **0.90** | **2.37** |

#### Continuous-route zone (bias)

| Route | n | Champion | Ex-ante | Ex-post |
|---|---|---|---|---|
| bird | 264 | −$1.49M | −$0.85M | **−$0.31M** |
| capspace | 69 | −$0.20M | +$0.16M | +$0.06M |
| other_cont | 210 | +$0.03M | +$0.32M | +$0.13M |

#### C2 signing-mechanism bias

| Mechanism | n | Champion | Ex-ante | Δ |
|---|---|---|---|---|
| Bird Rights | 274 | −$2.44M | −$1.80M | +$0.64M |
| Cap Space | 76 | −$0.23M | +$0.13M | +$0.36M |
| Early Bird | 46 | −$1.71M | −$1.34M | +$0.37M |
| MLE | 123 | +$0.59M | +$0.94M | +$0.35M |
| Minimum | 242 | +$1.97M | +$2.14M | +$0.17M |
| Non-Bird | 20 | +$2.06M | +$2.35M | +$0.29M |
| Sign & Trade | 13 | −$4.52M | −$4.20M | +$0.32M |
| Unknown | 131 | +$0.38M | +$0.68M | +$0.30M |

The δ shifts the entire surface upward, not just the Bird segment.
Bird's bias improves (−$2.44M → −$1.80M) but non-targeted segments drift:
MLE moves from +$0.59M to +$0.94M, Non-Bird from +$2.06M to +$2.35M.

#### C1 calibration

| Metric | Champion | Ex-ante |
|---|---|---|
| Calibration slope | 0.9849 | 0.9767 |
| \|slope − 1\| | 0.0151 | 0.0233 |
| Growth | — | **+0.0083** |

## Gate verdicts

| Gate | Criterion | Ex-ante | Verdict |
|---|---|---|---|
| 1. ΔSel t > 2 | t = 0.90 | — | **FAIL** |
| 2. Source control | N/A — δ from residuals, no external data | — | PASS (trivially) |
| 3. B1 forward drop ≤ 0.003 | −0.0005 | — | PASS |
| 4. C2 \|bias\| growth ≤ $0.3M | Bird +$0.64M, MLE +$0.35M | — | **FAIL** |
| C1 relative | \|slope−1\| growth +0.0083 > 0.005 | — | **FAIL** |

**The ex-ante form fails gates 1, 4, and C1.** Not adoptable.

The ex-post oracle passes gate 1 (t = 2.37) and closes Bird bias from
−$2.44M to −$1.55M, demonstrating that the δ idea itself is valid — the
loss is in P(route), not in δ.

## Anomalies

1. **72 high-salary rows in other_cont.** These are Unknown + MLE-labeled
   rows that the route-class priority puts into "continuous" because their
   salary does not land near an MLE exception amount (the MLE class uses a
   2% tolerance on exact USD amounts). They are correctly classified by the
   salary-landing rule, but the signing_cat label says MLE or Unknown, so
   they end up in other_cont rather than a more informative bucket.

2. **C2 upward drift on non-targeted segments.** The positive δ_bird (+$1.6M)
   is applied proportionally by P(bird), which is nonzero on all rows (mean
   P(bird) = 0.29 overall). Rows that are NOT bird get a positive push from
   P(bird) × δ_bird without a corresponding negative correction, shifting
   the whole surface up. This is the source of both the C2 growth and the
   calibration slope degradation.

3. **The ex-ante ΔSel (+0.00283) looks like a real signal diluted by the
   calibration damage.** The per-fold pattern (−0.002, +0.011, +0.006,
   +0.006, −0.007) is noisy but the three positive folds are larger than
   the two negatives. The slope degradation (0.985 → 0.977) means the
   model is systematically pushing predictions toward the center, which
   adds variance on every row — the δ benefit for Bird is real but drowned
   by the collateral damage.

## Mechanism

The retention premium (δ_bird = +$1.61M) is the **incumbent advantage** the
model cannot price: Bird Rights re-signings pay a premium for continuity
that the features cannot see (no team-history feature, ISSUES #5). The δ
recovers this as a population-level intercept correction. The fact that
δ_capspace and δ_other are essentially zero confirms that the gap is Bird-
specific, not a general continuous-route artifact.

## Files touched

- `docs/briefs/2026-07-26-route-delta.RESULT.md` (this file)
- Experiment script in scratchpad only — no production files modified.

## Proposed commit message

```
Analysis: per-route δ for the continuous routes (Bird vs cap space)

Split the continuous route class into bird / capspace / other, trained a
6-class route classifier, and estimated fold-honest per-route intercept
corrections (δ_k = mean OOF residual by route).

δ_bird = +0.012 cap_pct (+$1.61M) — stable across all 5 folds. The
ex-post oracle (true route) passes gate 1 at t=2.37 and closes Bird
bias from −$2.44M to −$1.55M. The ex-ante form (P-weighted) fails
gate 1 (t=0.90) and C1 calibration (+0.008 slope growth) because
P(bird)'s nonzero baseline on all rows shifts the whole surface.

Extension vs re-sign inside Bird: diff −$1.40M ± $1.50M, |diff|/SE < 1
— one δ for the Bird class. The loss is in P(route), not in δ.
```

## Explicit assertions

- **No route label enters FEATURE_COLS.** FEATURE_COLS = 14 features
  (regression). The classifier uses 34 features (14 + 20 enrichment
  columns). P(route) is an output composition weight, never a regression
  input.
- **No held-out row's residual enters its own δ.** Each fold's δ is
  computed from the other folds' OOF residuals (the champion's OOF
  predictions are themselves fold-honest).

## Recommendation

**Do not adopt the ex-ante form.** The δ idea is valid (ex-post t=2.37)
but the ex-ante delivery fails because P(bird) bleeds δ_bird onto non-Bird
rows, damaging calibration and shifting non-targeted segments. The gap
between ex-ante (t=0.90) and ex-post (t=2.37) localizes the loss to
P(route) precision — specifically, to the fact that P(bird) averages 0.22
even on non-Bird continuous rows, which is enough to propagate δ_bird
(+$1.6M) as a ~$0.35M positive shift everywhere.

**What would make it work:**

1. A **team-continuity feature** (ISSUES #5) would sharpen P(bird) and
   reduce cross-contamination. The signing-dates data already identifies
   same-team re-signings; if a binary same_team flag entered the classifier,
   P(bird) on non-Bird rows should fall.
2. A **zero-sum constraint** (forcing Σ_k n_k × δ_k = 0 across continuous
   routes) would eliminate the surface shift by construction, at the cost
   of pushing δ_capspace and δ_other negative. This is mechanically sound
   but would need its own battery run.
3. **Applying δ only to the ex-post (told-parameters) product**: when the
   route is told, δ_bird = +$1.6M is a legitimate price adjustment. This
   requires no classifier and no gate — it is the "given that he re-signed
   with Bird Rights, what is the price?" use case.

## ISSUES.md additions

None — the team-continuity data gap is already ISSUES #5, and the
signing-mechanism label issues are already ISSUES #4. No new issues
discovered.
