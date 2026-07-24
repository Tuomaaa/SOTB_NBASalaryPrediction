# RESULT — re-judging the gated max branch on corrected labels (phase 3)

**Brief**: `docs/briefs/2026-07-26-route-mixture-p3.md`
**Pin**: `master` @ e7a6336 (v7.12x line, 944 rows, max zone n=68, MAE $4.44M).
Worktree branch `route-mixture-p3`.
**Champion reproduction**: `fold_r2_selection` max|diff| vs the stored suite =
**0.00e+00**, so every delta below is candidate-minus-champion on identical
(fold, seed) cells.

**Verdict: DO NOT ADOPT. No cell passes.** The ISSUES #19 correction is real and
large — it lifts P(max) AUC from 0.9653 to **0.9823** and turns phase 2's
$0.02M honest ceiling into as much as $1.22M — but the branch still fails, and
now it fails in two *different* places depending on which τ rule is used:

- **τ\* cells (zero collateral)** sweep every brake and guard and fail the
  **Win** gate. Their honest ceilings — $0.15M (base) and $0.43M (enriched) —
  were already **below the $0.50M bar before a single arm ran**.
- **τ₉₀ cells** clear the Win gate ($1.12M / $0.83M) and fail **both brakes**
  (counterweight +$0.62M / +$0.36M, 25%+ band +$1.50M / +$0.82M), with base τ₉₀
  also failing B1 (+0.0115).

And the closest cell's win does not survive inspection. `enriched_τ*` realizes
+$0.42M against the $0.50M bar, but **+$1.58M of that is on the 12 locked
confirmation rows and only +$0.17M on the 56 selection rows** — and 48% of it is
one player, John Wall 2019, whom phase 2 already identified as a post-injury
supermax the market mispriced. The single number the architect should carry
forward: on the rows a decision is allowed to read, the best zero-collateral
cell wins **$0.17M**, a third of the bar.

---

## 1. What changed and why

Nothing shipped changes. The deliverable is one new evidence harness,
`scripts/eval_route_mixture_p3.py`, which re-runs phase 2's experiment on the
corrected v7.12x frame under the re-registered τ rule. `route_mixture.py` is
untouched — the `tau` gate and `attach_clf_features` landed in phase 2 already
do everything phase 3 needs, and the flag stays default **off**.

The harness differs from `eval_route_mixture_p2.py` in four ways, all of them
about evidence rather than mechanism. It fits the champion latent and **both**
classifiers once per (fold, seed) and derives every cell analytically from the
cached pieces, so all four cells are scored on bit-identical folds, seeds,
latents and probabilities and the P used to *choose* τ is the same P the branch
*runs* on. It reports the 25%+ brake in both the fixed-row and phase-2 own-band
readings, and C2 in both the "|bias| growth" and "signed change" readings,
because those pairs are not the same number and phase 2 silently picked one of
each (§8, anomalies 4 and 5). It splits the Win gate by the confirmation lock,
which turned out to matter more than anything else here. And it prints a
ceiling-vs-purity diagnostic so the architect can see the whole trade curve
without anyone reading a τ off a zone metric.

---

## 2. Part 1 — the classifiers on corrected labels

10-seed fold-honest OOF, 944 rows, 68 true maxes. The phase-2 rows are quoted
from that RESULT for contrast; they were measured on the pre-#19 labels.

| classifier | P(max) AUC | median P on true max | non-max P>0.5 | P(floor) AUC | P(mle) AUC |
|---|---:|---:|---:|---:|---:|
| phase-2 pin, base | 0.9653 | 0.5159 | 14 | 0.8253 | 0.7123 |
| phase-2 pin, enriched | 0.9557 | 0.4169 | 13 | 0.8243 | 0.7019 |
| **v7.12x base** (FEATURE_COLS) | **0.9823** | **0.8104** | 7 | 0.8259 | 0.7107 |
| **v7.12x enriched** (+20 batch cols) | 0.9784 | 0.7411 | 6 | 0.8246 | 0.6968 |

**The #19 fix, not the enrichment, is what sharpened the router.** Base AUC
0.9653 → 0.9823 and the median P on a true max 0.5159 → 0.8104 come entirely
from correcting `max_eligible_pct`: mis-tiered rows were previously labelled
"max" while looking nothing like one, and the classifier was being asked to fit
noise. Phase 2's finding that **the enrichment lowers P(max) AUC survives the
correction** (0.9823 → 0.9784, median 0.8104 → 0.7411) — the enriched set stays
a *variant*, never the default, exactly as the brief instructed.

Calibration by predicted P — enriched buys one thing, a clean top bin:

```
BASE                                    ENRICHED
[0.0,0.1)  n=846 meanP 0.006 emp 0.012  [0.0,0.1)  n=844 meanP 0.006 emp 0.009
[0.1,0.3)  n= 31 meanP 0.176 emp 0.129  [0.1,0.3)  n= 32 meanP 0.169 emp 0.250
[0.3,0.5)  n= 11 meanP 0.393 emp 0.455  [0.3,0.5)  n= 17 meanP 0.374 emp 0.412
[0.5,0.7)  n= 13 meanP 0.591 emp 0.692  [0.5,0.7)  n= 11 meanP 0.590 emp 0.818
[0.7,0.9)  n= 15 meanP 0.808 emp 0.867  [0.7,0.9)  n= 16 meanP 0.799 emp 0.750
[0.9,1.0)  n= 28 meanP 0.967 emp 0.964  [0.9,1.0)  n= 24 meanP 0.966 emp 1.000
```

**Purity curve** (cumulative true-max rate among OOF rows with P ≥ τ) — the
object both τ rules read, fixed before any zone metric:

```
 tau |  BASE n  nMax  purity |  ENR n  nMax  purity
0.50 |     56    49   0.875  |    51    45   0.882
0.54 |     52    47   0.904  |    50    44   0.880
0.58 |     51    46   0.902  |    48    42   0.875
0.62 |     47    44   0.936  |    42    37   0.881
0.64 |     47    44   0.936  |    40    36   0.900
0.70 |     43    40   0.930  |    40    36   0.900
0.74 |     41    38   0.927  |    37    34   0.919
0.78 |     38    36   0.947  |    33    30   0.909
0.80 |     35    34   0.971  |    31    28   0.903
0.84 |     33    32   0.970  |    30    27   0.900
0.86 |     31    30   0.968  |    27    24   0.889
0.88 |     28    27   0.964  |    24    24   1.000
0.90 |     28    27   0.964  |    24    24   1.000
0.94 |     21    20   0.952  |    20    20   1.000
0.95 |     20    20   1.000  |    18    18   1.000
```

The structural anti-correlation phase 2 identified is **still there and slightly
stronger**: Spearman(P, |champion error|) over the 68 maxes is **−0.761** (base)
and **−0.715** (enriched), against phase 2's −0.681. What changed is not the
sign of the relationship but the amount of error sitting at moderate P — which
is why the honest ceilings moved by 40× while the mechanism did not.

**Champion references, fixed v7.12x rows** (recomputed, not carried over):

| | value |
|---|---|
| A1 / A2 / B1 | 0.7878 / 0.8335 / 0.8303 |
| true-max zone (n=68) | MAE **$4.44M**, bias −$4.40M |
| — selection rows (n=56) | MAE $4.67M |
| — confirmation rows (n=12) | MAE $3.41M |
| counterweight [0.70,0.90) non-max (n=34) | bias −$4.87M, MAE $7.19M |
| 25%+ predicted band (n=52) | bias +$0.91M |

---

## 3. Part 1b — the two operating points, and their ceilings

Both rules read the purity curve only. τ\* is the brief's re-registered rule;
τ₉₀ is the old one, reported for contrast. Ceilings are computed **before** any
arm is scored.

| classifier | point | τ | purity | n ≥ τ | true max | non-max | touched | **honest ceiling** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | **τ\*** | 0.95 | 1.000 | 20 | 20 | 0 | 20/68 | **$0.147M** |
| base | τ₉₀ | 0.54 | 0.904 | 52 | 47 | 5 | 47/68 | $1.224M |
| enriched | **τ\*** | 0.88 | 1.000 | 24 | 24 | 0 | 24/68 | **$0.426M** |
| enriched | τ₉₀ | 0.64 | 0.900 | 40 | 36 | 4 | 36/68 | $0.849M |

The τ₉₀/enriched row reproduces the architect's re-run exactly (τ=0.64, 36/68,
ceiling $0.85M), which is the cross-check that this harness measures the same
thing phase 2's did.

**Both τ\* ceilings sit below the $0.50M win bar.** That is knowable from
probability space plus the champion's errors alone, before any arm runs, and it
is the whole result: on this frame, *zero collateral and a $0.50M win are not
simultaneously available.*

The brief's fallback ("highest-purity τ if 1.000 is unattainable") never fired —
both classifiers reach 1.000 at or below 0.95.

**Base does not beat enriched at τ\*** (ceiling $0.147M vs $0.426M, realized
$0.12M vs $0.42M), so the brief's "prefer base, fewer moving parts" branch does
not apply. Base's higher AUC buys nothing here because base never reaches purity
1.000 until τ=0.95, where only 20 maxes — the 20 the champion already prices —
remain above the gate.

---

## 4. Part 2 — the four-cell battery

Ship form is `push_clip_gated`; `hard_gated` is reported for every cell because
it is free once the fits exist. All rows are candidate-minus-champion on fixed
v7.12x rows.

| cell / arm | maxMAE | ΔMAE | **win** | **win (sel rows)** | cwBias | Δcw | Δ25 fixed | Δ25 own | ΔSel ± SE | t | A1 | A2 | B1 drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| champion | 4.44 | — | — | — | −4.87 | — | — | — | — | — | 0.7878 | 0.8335 | — |
| base_τ\* push_clip | 4.32 | −0.124 | +0.12 | +0.15 | −4.87 | +0.00 | +0.16 | +0.14 | +0.00030 ± 0.00022 | +1.38 | 0.7881 | 0.8345 | −0.0001 |
| base_τ\* hard | 4.32 | −0.124 | +0.12 | +0.15 | −4.87 | +0.00 | +0.16 | +0.14 | +0.00030 ± 0.00022 | +1.38 | 0.7881 | 0.8345 | — |
| base_τ₉₀ push_clip | 3.33 | −1.119 | +1.12 | +0.76 | −4.25 | **+0.62** | **+1.50** | **+1.28** | −0.00246 ± 0.00251 | −0.98 | 0.7881 | 0.8339 | **+0.0115** |
| base_τ₉₀ hard | 3.31 | −1.138 | +1.14 | — | −4.18 | **+0.69** | **+1.63** | **+1.39** | −0.00413 ± 0.00390 | −1.06 | 0.7855 | 0.8330 | — |
| **enr_τ\* push_clip** | 4.02 | −0.420 | **+0.42** | **+0.17** | −4.85 | +0.01 | +0.30 | +0.29 | −0.00010 ± 0.00052 | −0.20 | 0.7906 | 0.8346 | +0.0000 |
| enr_τ\* hard | 4.02 | −0.420 | +0.42 | +0.17 | −4.85 | +0.01 | +0.30 | +0.29 | −0.00010 ± 0.00052 | −0.20 | 0.7906 | 0.8346 | — |
| enr_τ₉₀ push_clip | 3.61 | −0.829 | +0.83 | +0.54 | −4.51 | **+0.36** | **+0.82** | **+1.08** | −0.00245 ± 0.00192 | −1.27 | 0.7893 | 0.8346 | +0.0027 |
| enr_τ₉₀ hard | 3.61 | −0.834 | +0.83 | — | −4.48 | **+0.39** | **+0.83** | **+1.14** | −0.00292 ± 0.00193 | −1.51 | 0.7891 | 0.8344 | — |

Bold = breaches its bar. `win` is the pooled 68-row zone-MAE improvement the
brief specifies; `win (sel rows)` is the same quantity over the 56 zone rows
outside the confirmation lock (§8, anomaly 1).

At τ\* the soft push and the hard switch **coincide exactly** for both
classifiers — at P ≥ 0.88 the push already lands every touched row on its
ceiling. They separate only at τ₉₀, where the hard arm wins marginally more zone
MAE and pays for it in both brakes and in ΔSel.

**Per-fold ΔSel** (selection pool, the decision matrix):

```
base_τ*      push_clip  [+0.00112  +0.00038  +0.00000  +0.00000  +0.00000]
base_τ90     push_clip  [-0.01074  +0.00053  +0.00084  +0.00274  -0.00564]
enriched_τ*  push_clip  [+0.00137  -0.00190  +0.00000  +0.00000  +0.00000]
enriched_τ90 push_clip  [-0.00103  -0.00942  +0.00000  +0.00160  -0.00339]
```

Three of five folds are exactly zero at τ\* for both classifiers: the gate
touches no selection-pool row in those folds at all.

**C2 fixed-segment integrity.** No cell breaches under the brief's wording
(|bias| growth); the largest is +$0.026M, on Unknown. Under phase 2's signed
reading the numbers are much larger because the push moves Bird Rights' bias
*toward zero* (−$2.44M → −$2.03M at base τ₉₀), i.e. the "growth" is a bias
*reduction*: max |signed change| is +$0.41M (base τ₉₀, would breach), +$0.27M
(enriched τ₉₀, matches the architect's quoted +0.27), +$0.09M (enriched τ\*),
+$0.03M (base τ\*). No verdict turns on which reading is used.

**B1 forward** (rolling-origin 2024-26, champion 0.8303):

```
                 forward R2   drop     origin 2024 / 2025 / 2026 max-zone MAE
champion            0.8303      —          $2.98M / $1.34M / $11.80M
base_τ*             0.8304  -0.0001        $2.92M / $1.31M / $11.80M
base_τ90            0.8188  +0.0115        $1.32M / $0.84M / $10.65M   <- FAIL
enriched_τ*         0.8303  +0.0000        $2.55M / $1.31M / $11.80M
enriched_τ90        0.8276  +0.0027        $1.71M / $1.19M / $11.80M
```

base_τ₉₀'s B1 failure is instructive: it *improves* every origin's max-zone MAE
and still drops forward R² by 0.0115, because at τ=0.54 the gate reaches
non-max rows whose damage outweighs the zone gain — origin 2025 alone falls
0.8130 → 0.7716.

---

## 5. Gate verdicts

| # | gate | bar | base_τ\* | base_τ₉₀ | **enr_τ\*** | enr_τ₉₀ |
|---|---|---|---|---|---|---|
| 1 | **Win**: true-max zone MAE improvement | ≥ $0.50M | ❌ +0.12 | ✅ +1.12 | ❌ **+0.42** | ✅ +0.83 |
| 1b | *Win on selection rows only* | *≥ $0.50M* | *❌ +0.15* | *✅ +0.76* | *❌ **+0.17*** | *✅ +0.54* |
| 2a | **Brake**: counterweight signed-bias growth | ≤ +$0.30M | ✅ +0.00 | ❌ +0.62 | ✅ **+0.01** | ❌ +0.36 |
| 2b | **Brake**: 25%+ band bias growth (fixed rows) | ≤ +$0.30M | ✅ +0.16 | ❌ +1.50 | ✅ **+0.30** | ❌ +0.82 |
| 2b′ | *same, phase-2 own-band form* | *≤ +$0.30M* | *✅ +0.14* | *❌ +1.28* | *✅ +0.29* | *❌ +1.08* |
| 3a | **Guard**: ΔSel t | > −2 | ✅ +1.38 | ✅ −0.98 | ✅ **−0.20** | ✅ −1.27 |
| 3b | **Guard**: C2 fixed-segment \|bias\| growth | ≤ $0.30M | ✅ +0.00 | ✅ +0.03 | ✅ **+0.03** | ✅ +0.03 |
| 3c | **Guard**: B1 forward drop | ≤ 0.003 | ✅ −0.0001 | ❌ +0.0115 | ✅ **+0.0000** | ✅ +0.0027 |
| | **overall** | | **fail (Win)** | **fail (brakes, B1)** | **fail (Win)** | **fail (brakes)** |

`enr_τ*` fails on one gate only, and by $0.08M pooled — but gate 1b says the
margin on decidable rows is $0.33M, not $0.08M. Note also that its 25%+ brake
lands at +$0.297M against a +$0.30M bar: it passes by $0.003M, which is inside
any reasonable notion of noise. Two of its six gates are effectively at the bar.

---

## 6. Part 3 — collateral, and the rows the adoption would buy

**Collateral (non-max rows above τ, ship form).**

τ\* is collateral-free for *both* classifiers, by construction — that is the
rule working, and it excludes both known brake-killers.

| cell | rows | total damage |
|---|---:|---:|
| base_τ\* (0.95) | **0** | $0.00M |
| enriched_τ\* (0.88) | **0** | $0.00M |
| enriched_τ₉₀ (0.64) | 4 | $18.34M |
| base_τ₉₀ (0.54) | 5 | $24.46M |

```
enriched_τ90, τ=0.64                       base_τ90, τ=0.54
player            P      damage            player            P      damage
jaylen brown '20  0.880   +0.00            jaylen brown '20  0.950   +0.00
jalen brunson '25 0.866   +2.50            austin reaves '26 0.799   +5.61
lamarcus aldr '19 0.862  +10.94            jalen brunson '25 0.774   +2.50
austin reaves '26 0.739   +4.90            james harden  '22 0.600   +9.76
                                           james harden  '25 0.595   +6.59
```

The brief named Aldridge 2019 and Reaves 2026 as the brake-killers; both are
excluded at τ\* under both classifiers, and neither was special-cased. Note the
two classifiers disagree about *which* non-max rows are dangerous — Aldridge is
P 0.862 enriched but only 0.409 base, while Harden 2022/2025 are base-only
collateral. **Jaylen Brown 2020 is harmless collateral** in every cell that
contains him: the champion already predicts him at his ceiling and overpredicts
by +$3.85M, so the push cannot move him.

**Touched true maxes at `enriched_τ*` (the closest cell): 24 of 68.** Eighteen
of the 24 carry a champion error of essentially zero and contribute nothing.
The entire realized win is six rows:

| player | season | P | champ err | push err | gain |
|---|---:|---:|---:|---:|---:|
| john wall | 2019 | 0.963 | −$13.55M | $0.00M | **+$13.55M** |
| tyrese haliburton | 2024 | 0.936 | −$5.92M | −$0.00M | +$5.92M |
| kemba walker | 2019 | 0.902 | −$3.37M | −$0.00M | +$3.37M |
| devin booker | 2024 | 0.975 | −$3.04M | $0.00M | +$3.04M |
| donovan mitchell | 2025 | 0.950 | −$2.04M | −$0.00M | +$2.04M |
| lamelo ball | 2024 | 0.967 | −$0.05M | $0.00M | +$0.05M |
| *(18 further maxes)* | | ≥0.90 | ≈$0.00M | ≈$0.00M | ≈$0.00M |

Total $27.97M over a 68-row zone = $0.41M. **John Wall 2019 alone is 48% of
it**, and Wall, Kemba and both Mitchell rows are confirmation-split players.

**Marginal coverage — what τ\* leaves untouched.** The brief's expectation holds:
the biggest zone errors are unreachable at any purity-respecting τ.

```
player                seas  champErr   P_base   P_enr
trae young            2026   -$32.53    0.045    0.032
jaren jackson jr.     2026   -$28.42    0.016    0.004
myles turner          2022   -$21.39    0.010    0.014
michael porter jr.    2022   -$17.04    0.016    0.015
john wall             2019   -$13.55    0.850    0.963   <- the one exception
jalen williams        2026   -$12.31    0.381    0.569
tobias harris         2019   -$11.96    0.014    0.021
jaren jackson jr.     2022   -$11.18    0.033    0.014
```

Seven of the eight largest zone errors sit below P = 0.06 under both
classifiers. The lone reachable one is the row phase 2 flagged as a market
mispricing.

---

## 7. Why no τ satisfies both sides — the ceiling-vs-purity trade

Diagnostic only. It is printed so the architect can see the whole curve; **no τ
was read off it**, and no cell in §4 was chosen using it.

```
ENRICHED                                  BASE
 tau  nAbove nMax purity ceiling nColl     tau  nAbove nMax purity ceiling nColl
0.95     18   18  1.000   0.259    0      0.95     20   20  1.000   0.147    0
0.90     24   24  1.000   0.426    0      0.90     28   27  0.964   0.286    1
0.88     24   24  1.000   0.426    0      0.86     31   30  0.968   0.502    1
0.86     27   24  0.889   0.426    3      0.80     35   34  0.971   0.744    1
0.85     30   27  0.900   0.503    3      0.78     38   36  0.947   0.883    2
0.80     31   28  0.903   0.642    3      0.74     41   38  0.927   1.007    3
0.70     40   36  0.900   0.849    4      0.70     43   40  0.930   1.081    3
```

**Enriched, in one sentence:** the three highest-P non-max rows — Jaylen Brown
2020 (0.880), Brunson 2025 (0.866), Aldridge 2019 (0.862) — rank *above* the
next two winnable maxes, Markkanen 2024 (0.858, −$5.23M) and Beal 2022 (0.805,
−$9.49M), so purity 1.000 is pinned at τ=0.88 with a $0.426M ceiling while the
ceiling only reaches $0.50M at τ=0.85, three collateral rows later.

**Base is the more interesting shape, and it is a live question the architect
should own.** Base never reaches purity 1.000 below 0.95, so *both* registered
rules treat it badly: τ\* jumps to 0.95 (ceiling $0.147M) and τ₉₀ — a
"smallest τ" rule — falls all the way to 0.54 (5 collateral rows, $24.46M of
damage). But base's purity curve has a **broad 0.946–0.971 plateau across
τ ∈ [0.78, 0.94]** whose only non-max resident is Jaylen Brown 2020, the row
whose push damage is $0.00M. At τ=0.80 that plateau offers a $0.744M ceiling
with one harmless collateral row.

I am **not** proposing that as a candidate and did **not** score it. A rule that
would land there — e.g. "smallest τ with purity ≥ 0.95", which selects base
τ=0.80 — is still a probability-space rule and would respect the standing
prohibition, but I am reporting it *after* having seen this run's zone metrics,
so registering it now would be exactly the τ-from-zone-MAE move the brief
forbids. It needs to be registered by the architect and run cold. Two things
temper the prospect: the base plateau's win would still be measured against the
confirmation-split problem in §8, and at τ=0.80 the gate reaches 34 maxes,
which on the enriched cells was enough to move the counterweight brake.

---

## 8. Anomalies

**1 — The win lives in the confirmation split. This is the important one.**
The Win gate as specified reads a zone MAE pooled over all 68 zone rows, 12 of
which are locked confirmation players. Split:

```
                    zone MAE          win vs champion
                  all   sel   conf     all     sel     conf
champion         4.44  4.67   3.41       —       —        —
base_τ*          4.32  4.52   3.41   +0.12   +0.15    +0.00
base_τ90         3.32  3.90   0.62   +1.12   +0.76    +2.79
enriched_τ*      4.02  4.49   1.83   +0.42   +0.17    +1.58
enriched_τ90     3.61  4.13   1.21   +0.83   +0.54    +2.20
```

Every cell wins 2–10× more on the 12 canary rows than on the 56 decidable ones.
The same signal shows up in A1: `enriched_τ*` posts pooled A1 0.7906 against the
champion's 0.7878 (+0.0028) while its selection-pool paired ΔSel is **−0.0001**
(t = −0.20) — the entire pooled A1 gain is confirmation rows. The mechanism is
mundane and not a bug: the confirmation split is 15% of *players*, and it
happens to hold Wall, Kemba and both Mitchell rows, four of the six rows that
carry the τ\* win. But it means the Win gate as written is not a clean decision
metric, and a future candidate could clear it on canary rows alone. Filed as
ISSUES #20.

**2 — 48% of the τ\* win is John Wall 2019.** Phase 2 (§7) already called this
row out: a post-injury 35% supermax the market arguably overpaid and the model
*should* discount. Strip it and `enriched_τ*` realizes $0.21M pooled and
~$0.17M on selection rows (Wall is himself a confirmation row, so the selection
figure is unaffected). Concentration for the other cells: base_τ\* 63% in one
row (Haliburton), enriched_τ₉₀ 26%, base_τ₉₀ 18%.

**3 — The enrichment lowers AUC and is still the better τ\* classifier.** Same
paradox as phase 2, now with real coverage: enriched is worse on every
discrimination statistic (AUC 0.9784 vs 0.9823, median-on-true 0.741 vs 0.810)
yet is the only classifier reaching purity 1.000 below τ=0.95, so τ\* rewards it
(24 touchable maxes vs 20, ceiling $0.43M vs $0.15M). The τ\* rule is a
top-of-distribution rule and AUC is a whole-distribution statistic; they are
allowed to disagree, and here they do. Nothing to fix — but it is why "prefer
base, fewer moving parts" does not apply.

**4 — The 25%+ brake was measured on moving row groups in phase 2.** The
committed `eval_route_mixture_p2.py` compares the champion's own 25%+ band
against the *candidate's* own 25%+ band, which the worker brief forbids for
model-vs-model comparison. The candidate's band is larger by construction
(55 rows vs 52 at enriched τ\*, 63 vs 52 at base τ₉₀) because pushing rows up
moves them into the band. Both readings are reported above; the gap is material
in magnitude (enriched τ₉₀: +$0.82M fixed vs +$1.08M own — the architect's
quoted number is the own-band one) but **changes no verdict here**. Filed as
ISSUES #20.

**5 — C2's two readings differ by 15×.** "Signed change" and "|bias| growth"
diverge whenever a segment's bias moves toward zero, which is exactly what the
push does to Bird Rights (−$2.44M → −$2.03M at base τ₉₀). Max signed change
+$0.41M would breach the $0.30M bar; max |bias| growth is +$0.026M and does not.
The brief's wording is "|bias| growth", so that is what §5 gates on. No verdict
turns on it — base τ₉₀ fails three other gates regardless. Also in ISSUES #20.

**6 — Nothing scores implausibly well.** The largest delta anywhere is
base τ₉₀'s $1.12M zone win, which is fully accounted for by its 47 touched
maxes and is paid for immediately in both brakes and B1. The escalation trigger
("a result that passes the gates but whose mechanism you cannot explain in one
sentence") does not fire, because no result passes the gates.

---

## 9. Recommendation

**Do not adopt. Keep the flag default off.** No cell passes; the two rules fail
on opposite sides and the gap between them is not a tuning gap but the ordering
of Aldridge/Brunson/Brown against Markkanen/Beal in probability space.

If the architect wants one more attempt before closing the route-mixture line,
the single highest-value move is **not** another classifier: it is to register a
purity rule that can address the base classifier's [0.78, 0.94] plateau (§7) and
run it cold, *and* to fix the Win gate to read selection rows only (§8, anomaly
1) before doing so. On the numbers in §7 the best that rule could win is
$0.744M pooled — and if the confirmation pattern in §8 holds, roughly half that
on decidable rows, which puts it right back at the bar. My reading is that phase
2's structural conclusion was wrong in magnitude but right in direction: the
correction bought a real 40× in reachable error, and it still is not enough,
because the reachable error is concentrated in one mispriced contract and the
canary split.

Phase 2's other standing conclusions are untouched and were not re-litigated:
the enrichment stays a variant, the machinery stays landed and inert, and the
floor branch remains a NO-GO for the same left-bound reason.

---

## 10. Files touched, and where the branch lives

Branch **`route-mixture-p3`** (worktree `../BBall-worker-route-p3`, pinned to
`master` @ e7a6336).

- `scripts/eval_route_mixture_p3.py` (**new**) — the whole evidence harness.
  Reproduce with `OMP_NUM_THREADS=6 python scripts/eval_route_mixture_p3.py`
  (~7 min; champion reproduction is asserted against the stored suite).
- `docs/briefs/2026-07-26-route-mixture-p3.RESULT.md` (**new**) — this file.
- `ISSUES.md` — one new entry, #20.

Artifacts (gitignored `outputs/models/`, for review):
`route_mixture_p3_eval.json` (every number above) and
`route_mixture_p3_oof.csv` (per-row P under both classifiers, champion OOF,
all eight arm OOFs, and the forward predictions).

**Not touched**: `src/model/route_mixture.py`, `train.py`, `evaluate_suite.py`,
the target, the filter chain, `FEATURE_COLS`, sigma/gate/censor settings, and
the docs-agent lane. Frame unchanged at 944 rows; the champion reproduces the
stored reference exactly (max|diff| 0.00e+00).

---

## 11. Proposed commit message (architect edits and lands)

```
Route-mixture phase 3: re-judge the gated max branch on corrected labels

The ISSUES #19 fix is worth a great deal to the router and still not enough to
gate. On the corrected v7.12x labels P(max) AUC rises 0.9653 -> 0.9823 and the
median P on a true max 0.5159 -> 0.8104 (the enrichment, as in phase 2, LOWERS
both: 0.9784 / 0.7411). Phase 2's $0.02M honest ceiling becomes $0.15M-$1.22M
depending on tau. The structural anti-correlation survives: Spearman(P, |champ
err|) = -0.761 base, -0.715 enriched.

Four cells, {base, enriched} x {tau* = smallest tau with purity 1.000,
tau90 = the old >=90% rule}, ship form push_clip_gated plus hard_gated. NONE
passes, and they fail on opposite sides:

  base     tau*=0.95  ceiling $0.147M  win +0.12  all brakes/guards pass
  enriched tau*=0.88  ceiling $0.426M  win +0.42  all brakes/guards pass
  base     tau90=0.54 ceiling $1.224M  win +1.12  cw +0.62, 25band +1.50, B1 +0.0115
  enriched tau90=0.64 ceiling $0.849M  win +0.83  cw +0.36, 25band +0.82

Both tau* ceilings were below the $0.50M win bar BEFORE any arm ran: zero
collateral and a $0.50M win are not simultaneously available on this frame. The
mechanism is an ordering in probability space -- the three highest-P non-max
rows (Jaylen Brown '20 0.880, Brunson '25 0.866, Aldridge '19 0.862) rank above
the next winnable maxes (Markkanen '24 0.858, Beal '22 0.805). tau* excludes
both named brake-killers (Aldridge, Reaves) with no special-casing.

The closest cell does not survive inspection: enriched_tau*'s +$0.42M is
+$1.58M on the 12 locked confirmation rows and only +$0.17M on the 56 selection
rows, and 48% of it is John Wall 2019 -- the mispriced post-injury supermax
phase 2 already flagged. Every cell wins 2-10x more on canary rows than on
decidable ones; the Win gate as specified pools them.

Analysis only. route_mixture.py untouched, flag stays default off, champion
bit-identical (fold_r2_selection max|diff| 0.00e+00). No version number.
```

No version number claimed — nothing gated, and the assignment is the
architect's in any case.

---

## 12. ISSUES.md additions

One new entry, **#20** — three related evaluation-protocol defects found while
building this harness, none of which changes a verdict here but each of which
could change a future one: the Win/zone gate pools the confirmation split, the
committed phase-2 harness compares moving 25%+ row groups, and "C2 |bias|
growth" is implemented as a signed change. Written to be actionable cold.
