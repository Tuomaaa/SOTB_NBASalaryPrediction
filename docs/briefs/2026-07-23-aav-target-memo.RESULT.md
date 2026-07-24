# RESULT — AAV target-variable feasibility memo

**Task**: `docs/briefs/2026-07-23-aav-target-memo.md` · **Pin**: `v7.8x` ·
**Branch**: `aav-target-memo` · **Date**: 2026-07-23

Analysis only. Nothing was adopted, no pipeline file was modified, and no
version number is claimed. There are no gate verdicts because nothing is being
proposed for adoption.

**Recommendation up front: REJECT the switch.** The premise does not survive
the check that the target is measured in *cap share*, not dollars. Details in
§5.

---

## 1. What was done and why

The brief asks whether AAV would be a better target than year-1 pay, on the
theory that front- and back-loaded deals put different year-1 numbers on
identical total commitments and thereby inject noise the model cannot learn. I
matched Spotrac contracts to the 1,172-row v7.8x evaluation frame under an
explicit legality rule, measured how far AAV sits from year-1, wrote down what
each Stage-2 bound would have to become, and refit the champion on the covered
subset with AAV as the target.

The divergence is real and large in nominal dollars — a five-year maximum
contract's AAV is exactly 16% above its year-1 salary, $8.7M on Jayson Tatum's
2025 deal. It is almost entirely **not** noise, and almost entirely **not**
present in the units the model actually works in. It is not noise because it is
a deterministic function of contract length and raise rate, `aav/y1 = 1 +
r(n−1)/2`, with `r` taking a handful of legal values. And it largely vanishes in
cap-share terms because the later years of a contract are paid against a larger
cap: escalators run at 5–8% of year-1 salary while the cap itself compounded at
6.17% a year over 2019–2026, so the two nearly cancel. Nominal AAV divided by
the *signing-year* cap is not a cap share of anything.

---

## 2. Coverage

### The matching rule, and the tolerance it rests on

A Spotrac contract explains an evaluation row when all three hold:

1. **it starts in the row's season** (`contract_start == season`). The row wears
   a year-1 label, so a contract starting earlier is a continuation, not the
   deal that set this price;
2. **its length is legal** (1–5 years) — see the 10-day parse bug in §Anomalies;
3. **its total value reconciles with the row's year-1 pay under a legal
   escalator.** CBA raises are a fixed percentage `r` of year-1 salary, so

   ```
   total = y1 · (n + r·n(n−1)/2)     ⟹     aav/y1 = 1 + r(n−1)/2
   ```

   and the implied `r` must sit inside the legal band.

**The tolerance is stated on the raise rate, not on |AAV − year-1|** — the
latter is the quantity this memo exists to measure, and a tolerance on it would
delete the finding. I used **|r| ≤ 0.10** against a legal band of ±0.08, the
slack absorbing rounding and incentive money; for one-year deals, where `r` is
undefined, **|total/y1 − 1| ≤ 0.10**. Among multiple candidates the smallest
deviation wins.

The rule validates itself. The implied `r` over start-anchored candidates piles
up on precisely the legal escalator rates:

| implied `r` | 0.080 | 0.050 | 0.000 | −0.080 | −0.050 | everything else |
|---|---|---|---|---|---|---|
| contracts | 127 | 125 | 74 | 12 | 9 | scattered |

89.3% of multi-year candidates land inside ±0.09 with no tuning.

### Coverage: 604 / 1,172 = 51.5%

| reason | rows |
|---|---|
| matched | **604** |
| no start-anchored contract | 396 |
| value does not reconcile | 172 |

**By season** — note this is *not* a time trend, which matters for the brief's
2019 trap:

| season | n | matched | no contract | no reconcile | coverage |
|---|---|---|---|---|---|
| 2019 | 220 | 97 | 110 | 13 | 44.1% |
| 2020 | 148 | 69 | 54 | 25 | 46.6% |
| 2021 | 159 | 108 | 44 | 7 | **67.9%** |
| 2022 | 116 | 55 | 33 | 28 | 47.4% |
| 2023 | 143 | 72 | 42 | 29 | 50.3% |
| 2024 | 132 | 67 | 36 | 29 | 50.8% |
| 2025 | 131 | 67 | 38 | 26 | 51.1% |
| 2026 | 123 | 69 | 39 | 15 | 56.1% |

2019 is the weakest season, as warned — but 2021 is the *strongest* at 67.9%
and 2022 falls back to 47.4%. The pattern is not monotone in time, so this is
not the 2019-continuation artifact wearing a new hat. It tracks mechanism mix:

| mechanism | n | matched | coverage |
|---|---|---|---|
| Early Bird | 48 | 42 | 87.5% |
| Bird Rights | 309 | 251 | 81.2% |
| Sign & Trade | 15 | 12 | 80.0% |
| Non-Bird | 22 | 13 | 59.1% |
| MLE | 182 | 105 | 57.7% |
| Cap Space | 117 | 61 | 52.1% |
| **Minimum** | 303 | 106 | **35.0%** |
| Other | 41 | 14 | 34.1% |
| Unknown | 132 | 0 | 0.0% |

**By salary band** — the bias that matters:

| year-1 salary | n | matched | coverage |
|---|---|---|---|
| < $3M | 530 | 142 | **26.8%** |
| $3–8M | 216 | 149 | 69.0% |
| $8–15M | 189 | 132 | 69.8% |
| $15–25M | 121 | 89 | 73.6% |
| $25M+ | 116 | 92 | **79.3%** |

### The covered subset is a rich-player slice

| | covered (604) | uncovered (568) |
|---|---|---|
| salary | **$12.37M** | **$5.52M** |
| `cap_pct` | 0.094 | 0.045 |
| `mpg` | 23.6 | 18.6 |
| `darko_dpm_z` | +0.153 | −0.348 |
| `prev_cap_pct` | 0.069 | 0.037 |
| share max-contract rows | 8.3% | 1.2% |
| share at-floor rows | **17.5%** | **33.6%** |

This is the collection-artifact channel the worker brief's gate 2 exists for.
Every number in §3–§4 is conditioned on a subset that over-represents exactly
the players whose contracts diverge most and under-represents the floor zone by
half. **A switch estimated here would be estimated on the top half of the
market.**

---

## 3. Divergence

Over the 604 covered rows, with `ratio = AAV/year-1 = aav_cap_pct / cap_pct`:

| threshold | rows | share |
|---|---|---|
| differ by > 2% | 410 | 67.9% |
| differ by > 5% | 244 | 40.4% |
| differ by > 10% | 107 | 17.7% |
| differ by > 15% | 43 | 7.1% |
| differ by > 20% | **0** | 0.0% |
| **equal within 0.5%** | **178** | **29.5%** |

Direction: AAV above year-1 on 358 rows, below on 68, equal on 178. The
brief's third trap is confirmed — **AAV ≥ year-1 does not hold**; 62 rows carry
a declining structure with mean ratio 0.9421.

Gap size: mean **+$0.769M**, sd $1.726M, mean absolute **$0.966M**
(= 0.729 percentage points of cap).

### Which contracts diverge

Entirely explained by length and raise rate, exactly as `1 + r(n−1)/2` predicts:

| contract years | n | mean ratio | mean \|gap\| | share > 5% |
|---|---|---|---|---|
| 1 | 126 | 1.0044 | $0.02M | 11.9% |
| 2 | 167 | 1.0173 | $0.18M | 0.0% |
| 3 | 129 | 1.0325 | $0.66M | 60.5% |
| 4 | 127 | 1.0569 | $1.69M | 79.5% |
| 5 | 55 | **1.1260** | **$4.57M** | 90.9% |

| segment | n | mean ratio | mean \|gap\| | share > 5% |
|---|---|---|---|---|
| max contracts (≥90% of own ceiling) | 50 | **1.1218** | **$5.00M** | 94.0% |
| at the CBA floor | 106 | 1.0126 | $0.04M | 12.3% |
| everything else | 448 | 1.0321 | $0.73M | 41.1% |

| escalator direction | n | mean ratio | mean gap |
|---|---|---|---|
| declining (`r` < −0.01) | 62 | 0.9421 | −$0.94M |
| flat (\|`r`\| ≤ 0.01) | 77 | 1.0002 | +$0.00M |
| rising (`r` > 0.01) | 339 | 1.0732 | +$1.54M |
| one-year (no `r`) | 126 | 1.0044 | +$0.01M |

By mechanism, the divergence is concentrated in Bird Rights (mean ratio 1.0595,
mean gap +$1.63M) — the only mechanism that routinely buys five years — while
Minimum (1.0126) and MLE (1.0208) barely move.

### The 10 largest divergences

| player | season | year-1 | yrs | total | AAV | `r` | ratio | gap |
|---|---|---|---|---|---|---|---|---|
| Jayson Tatum | 2025 | $54.126M | 5 | $313.9M | $62.79M | 0.080 | 1.160 | **+$8.660M** |
| Jaylen Brown | 2024 | $49.206M | 5 | $285.4M | $57.08M | 0.080 | 1.160 | +$7.873M |
| Nikola Jokić | 2023 | $47.607M | 5 | $276.1M | $55.22M | 0.080 | 1.160 | +$7.617M |
| Cade Cunningham | 2025 | $46.394M | 5 | $269.1M | $53.82M | 0.080 | 1.160 | +$7.423M |
| Evan Mobley | 2025 | $46.394M | 5 | $269.1M | $53.82M | 0.080 | 1.160 | +$7.423M |
| Bradley Beal | 2022 | $43.279M | 5 | $251.0M | $50.20M | 0.080 | 1.160 | +$6.925M |
| Tyrese Haliburton | 2024 | $42.176M | 5 | $244.6M | $48.92M | 0.080 | 1.160 | +$6.748M |
| Anthony Edwards | 2024 | $42.176M | 5 | $244.6M | $48.92M | 0.080 | 1.160 | +$6.748M |
| Jalen Williams | 2026 | $41.500M | 5 | $239.2M | $47.84M | 0.076 | 1.153 | +$6.339M |
| Chet Holmgren | 2026 | $41.500M | 5 | $239.2M | $47.84M | 0.076 | 1.153 | +$6.339M |

Every one is a five-year Bird-Rights maximum at the identical ratio 1.160. The
top ten is one contract type, not ten findings.

### Hand check against Spotrac

Checked against the **cached** player pages — the cache is what the pipeline
parsed, so checking it tests the parse, and re-fetching would spend the rate
limit for no extra information.

- **Jayson Tatum**, `jayson-tatum.html`: page block reads
  `Contract Terms: 5 yr(s) / $313,933,410`, `Average Salary: $62,786,682`,
  `Signed Using: Bird Rights`, `Free Agent: 2030 / UFA`.
  $313,933,410 / 5 = $62,786,682 ✓. Against the training row's year-1 pay:
  $54.126M × (5 + 0.08×10) = $313.93M ✓ — the escalator model reproduces
  Spotrac's total to the dollar.
- **Jaylen Brown**, `jaylen-brown.html`: `5 yr(s) / $285,393,640`,
  `Average Salary: $57,078,728`, `Free Agent: 2029 / UFA`.
  $49.206M × 5.8 = $285.39M ✓.
- **Nikola Jokić**, `nikola-jokic.html` (third-largest, checked as a bonus):
  `5 yr(s) / $276,122,630`, `$55,224,526`, `Free Agent: 2028 / UFA`.
  $47.607M × 5.8 = $276.12M ✓.

All three reconcile exactly. The divergence is arithmetic, not a data error.

---

## 4. Bounds interaction

**This section is the mechanical verdict, and it is not clean.**

Stage 2 clips into `[floor_pct, max_eligible_pct]`, both defined on year-1 pay.
Scoring an AAV target against those same numbers:

| | year-1 target | AAV target |
|---|---|---|
| covered rows above `max_eligible_pct` | 0 (see Anomalies) | **46 (7.6%)** |
| covered rows below `floor_pct` | 24 | 20 (3.3%) |

The 46 breaches are not marginal — every one sits at ratio **exactly 1.1600**
of its own ceiling (Giannis 2021, Beal 2022, Jokić 2023, Brown 2024, Tatum 2025
at 0.406 against a 0.35 ceiling; Edwards 2024 and Mobley 2025 at 0.348 against
0.30; Garland 2023 at 0.290 against 0.25). Under an AAV target these rows'
targets are **unreachable by construction** — the clip pins the prediction below
the truth, which METHODOLOGY's ceiling audit calls provably wrong and worth
fixing twice over.

### What each bound would have to become

**Ceiling.** From `aav/y1 = 1 + r(n−1)/2`, the AAV ceiling is
`max_eligible_pct · (1 + r(n−1)/2)`:

| n | r = −0.08 | −0.05 | 0.00 | +0.05 | +0.08 |
|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.960 | 0.975 | 1.000 | 1.025 | 1.040 |
| 3 | 0.920 | 0.950 | 1.000 | 1.050 | 1.080 |
| 4 | 0.880 | 0.925 | 1.000 | 1.075 | 1.120 |
| 5 | 0.840 | 0.900 | 1.000 | 1.100 | **1.160** |

**The multiplier is a function of contract length and raise rate, and neither
exists before the contract is signed.** The year-1 ceiling is knowable ex ante
from experience, awards and previous salary — that is what makes Stage 2 a
legitimate inference-time device. An AAV ceiling is not: to clip a prediction
you would first have to predict how many years and what escalator the parties
will agree to. Options are to take the loosest bound (×1.16, which stops
censoring almost everything and guts the max side), or to predict `n` and `r`
first (a second model, whose error enters the bound).

**Floor.** Worse, not mirror-image. Over the 106 covered at-floor rows,
`aav_cap_pct / floor_pct` runs from **0.5172 to 1.6222** (median 1.0095), with
19 rows below the floor. A minimum contract's later years step up the
service-year minimum scale, and multi-year minimums are commonly partly
non-guaranteed, so AAV wanders on both sides of a bound that year-1 pay sits
exactly on. `floor_pct` is already a recovered quantity (ISSUES #6) rather than
a published scale; restating it for AAV means recovering a *different* quantity
from mass points that no longer exist, because AAV does not pile up on the
minimum scale.

**Verdict: this is a Stage-2 redesign, not a rescale.** Both bounds lose the
property that makes them usable — being knowable before the deal is signed.

### The denominator problem, which decides the memo

Nominal AAV divided by the **signing-year** cap is not a cap share. Later years
are paid against a larger cap. Modelling `salary_t = y1·(1 + r·t)` and dividing
each year by *its own* cap gives the true mean cap share:

| all 604 covered rows | mean |
|---|---|
| nominal AAV / signing-year cap | 0.1000 |
| **cap-weighted mean share** | **0.0926** |
| year-1 `cap_pct` | 0.0942 |

Nominal AAV overstates the real cap burden by **+5.45%**. Relative to year-1,
nominal AAV sits **+3.61%** away while the cap-weighted share sits **−1.68%**
away — *the correctly-denominated AAV is closer to year-1 than nominal AAV is.*

Because 2027+ caps in `config.py` are projections, I re-ran this on the **466
contracts that finish inside realized cap data (≤ 2026)**:

| realized caps only | mean ratio to year-1 |
|---|---|
| nominal AAV / signing-year cap | 1.0282 |
| **cap-weighted true share** | **0.9851** |

| contract years | n | nominal | cap-weighted |
|---|---|---|---|
| 1 | 126 | 1.0044 | 1.0000 |
| 2 | 150 | 1.0172 | 0.9903 |
| 3 | 93 | 1.0314 | 0.9712 |
| 4 | 75 | 1.0527 | 0.9631 |
| 5 | 22 | **1.1430** | **0.9979** |

And on the max rows specifically (n = 25, realized caps only): nominal
**1.1236**, cap-weighted **1.0029**.

**A five-year maximum contract's average cap share is within 0.3% of its year-1
cap share.** The 16% divergence is an artifact of the denominator, not a
property of the contract. Realized cap CAGR 2019–2026 was 6.17% against
escalators of 5–8% of year-1 salary; the two cancel by construction of the CBA,
which sets both.

---

## 5. Trial fit — evidence, not adoption

604 covered rows, 14 features, 5-fold GroupKFold by player, 10 seeds, champion
hyperparameters.

### (a) The target's own noise proxy

| | value |
|---|---|
| sd(AAV − year-1) | 0.01292 cap pts = **$1.726M** |
| mean\|AAV − year-1\| | 0.00729 cap pts = **$0.966M** |
| sd(year-1 target) | 0.08561 |
| sd(AAV target) | 0.09532 |
| sd(gap) / sd(year-1 target) | 0.1509 |

The entire year-1-vs-AAV disagreement averages **$0.97M** against a champion CV
MAE of **$3.065M**. Even if all of it were noise and all of it were removable,
the ceiling on the prize is about a third of current error — and per §3 it is
not noise, it is `1 + r(n−1)/2`.

### (b) Headline and residual structure

| configuration | R² | MAE $M | MAE/sd | bias $M | calib slope | intercept |
|---|---|---|---|---|---|---|
| year-1, Grabit (champion form) | 0.7662 | **3.497** | 0.3172 | −0.041 | 0.9678 | +0.00332 |
| year-1, plain XGBoost | 0.7620 | 3.673 | 0.3337 | +0.029 | 1.0014 | −0.00050 |
| AAV, Grabit, year-1 bounds (naive port) | 0.7702 | 3.942 | 0.3201 | −0.228 | 1.0094 | +0.00078 |
| AAV, Grabit, **ORACLE** AAV bounds | 0.7747 | 3.816 | 0.3105 | −0.048 | 0.9729 | +0.00300 |
| AAV, plain XGBoost | 0.7708 | 3.994 | 0.3253 | +0.044 | 0.9983 | −0.00035 |

The ORACLE row uses the *realized* contract length and raise rate to build its
bounds. It is not a pipeline — it is an upper bound on what a solved-bounds AAV
model could reach, and it is reported only so the naive port's damage can be
separated from the target change itself.

**Do not read the R² column as a comparison.** sd(AAV) is 11% larger than
sd(year-1), so R²'s denominator grew — this is D1 logic applied to targets
rather than row sets. MAE moved the *opposite* way ($3.497M → $3.942M). On the
scale-free MAE/sd the two are within noise of each other (0.3172 vs 0.3201
naive, 0.3105 oracle).

C1 calibration, bias by predicted band ($M):

| configuration | <2% | 2-4% | 4-8% | 8-15% | 15-25% | 25%+ |
|---|---|---|---|---|---|---|
| year-1, Grabit | −0.79 | −0.37 | +0.30 | −0.31 | −0.18 | +1.98 |
| AAV, Grabit, naive | −0.82 | −0.35 | +0.14 | −0.05 | −0.65 | −0.43 |
| AAV, Grabit, oracle | −0.81 | −0.35 | +0.14 | −0.05 | +0.24 | +0.35 |

C2 mechanism bias ($M), **fixed row groups**:

| configuration | Bird (251) | Min (106) | MLE (105) | Cap Sp (61) | E.Bird (42) |
|---|---|---|---|---|---|
| year-1, Grabit | **−1.78** | +2.22 | +1.61 | +0.57 | −0.70 |
| AAV, Grabit, naive | **−2.44** | +2.33 | +1.79 | +0.88 | −0.64 |
| AAV, Grabit, oracle | −2.01 | +2.34 | +1.79 | +0.91 | −0.64 |

The naive port worsens Bird Rights bias by **$0.66M** — more than double the
$0.3M C2 gate — because Bird Rights is where the five-year deals live and the
year-1 ceiling makes their targets unreachable. Oracle bounds recover most but
not all of it (−$0.23M). Nothing gets structurally cleaner anywhere; the
residual signature is the same shape with slightly more of it.

Naive port mechanics: **30 rows** pinned exactly at the year-1 ceiling,
**46 rows** whose AAV target is above that ceiling and therefore unreachable.

### (c) The null-difference subset — the only clean anchor

178 rows where AAV equals year-1 within 0.5%. On these the two targets are
literally the same number, so scoring both models here removes the rescaling
and compares only what each training signal taught.

| configuration | R² | MAE $M | bias $M |
|---|---|---|---|
| **year-1, Grabit** | **0.0953** | **2.938** | **+1.562** |
| year-1, plain XGBoost | 0.0982 | 3.100 | +1.801 |
| AAV, Grabit, naive | −0.0381 | 3.149 | +1.891 |
| AAV, Grabit, oracle | −0.0381 | 3.149 | +1.891 |
| AAV, plain XGBoost | −0.0223 | 3.310 | +2.125 |

**Training on AAV makes predictions worse on the rows where AAV and year-1 are
the same number** — MAE +$0.21M, bias +$0.33M, R² from +0.095 to −0.038, and
the pattern repeats for the plain-XGBoost pair, so it is not a bounds artifact.
The mechanism is one sentence: the AAV-trained model learns the average upward
offset of multi-year deals and then applies it to the short deals that do not
have one.

Composition caveat, stated because the low absolute R² invites
misreading: the null subset is short cheap deals (mean 1.71 years, $6.26M,
`cap_pct` 0.048, against 3.11 years and $14.93M for the rest; 103 of 178 are
one-year deals). Low target variance is why R² is near zero for *every*
configuration. The comparison is between models on **fixed rows**, which is
valid; the absolute level is not a claim about model quality.

---

## 6. Recommendation — REJECT

Not a switch, and not a dual target.

**Reason 1 — the premise does not survive the units.** The model's target is
cap share, not dollars. Measured correctly, a five-year maximum contract's
average cap share is 1.0029 of its year-1 cap share (n=25, realized caps only),
and the whole covered set sits at 0.9851. The 16% gap that motivates the brief
comes from dividing all five years by the signing-year cap, which measures
nothing. The CBA sets both the escalator (5–8% of year-1) and the cap path
(6.17% realized CAGR), and they cancel. There is no noise here to remove.

**Reason 2 — the price is a Stage-2 redesign paid on half the data.** Both
bounds lose the property that makes Stage 2 legitimate — computability before
the deal is signed. The AAV ceiling needs contract length and raise rate; the
AAV floor is not even a bound (at-floor AAV ranges 0.52–1.62 of `floor_pct`).
And only 51.5% of rows can be given an AAV at all, with coverage running 79%
at the top of the market against 27% at the bottom and the floor zone
represented at half its true rate. The null-difference test says the switch is
not merely neutral but actively harmful on the rows that can adjudicate it.

**The main risk of this recommendation.** Rejecting the target change leaves a
real fact unmodelled: in nominal committed dollars, a five-year max deal *is* a
$8.7M-per-year larger commitment than its year-1 number suggests, and a reader
asking "what did this contract cost the team" is asking a fair question. That
question is **Contract Surplus**, not **Signing Residual** — a display-layer
computation over a known contract, not a target for a model that predicts
prices before contracts exist. If it is ever wanted, compute
`Σ_t salary_t / cap_t` from the contract table and show it next to the
prediction; that needs no retraining and no version break. The secondary risk
is that the cancellation is era-dependent: it holds while cap growth tracks the
escalator band. If cap growth fell to ~2% for several years, nominal and
cap-weighted AAV would diverge and this memo should be re-run. `config.py`'s
2027–2029 projections (+5%/yr) already sit below the realized 6.17%, which is
why §4's headline was re-derived on realized caps only.

---

## Anomalies

1. **AAV's R² is higher than year-1's, and it means nothing.** 0.7702 vs
   0.7662 looks like a win and is the D1 trap on the target axis: sd(AAV) is
   11% larger, so R²'s denominator grew. MAE moved the other way, $3.497M →
   $3.942M. Flagged loudly because this is precisely the number someone would
   quote to justify the switch.

2. **The top-10 divergence table is one contract type.** All ten are five-year
   Bird-Rights maxima at ratio 1.1600 (or 1.153 for the two 2026 rows on a
   7.6% escalator). Not a red flag — the mechanism is `1 + r(n−1)/2` and it is
   verified by hand — but it means "10 largest divergences" is one finding
   reported ten times, and the concentration is the finding.

3. **Spotrac 10-day contracts parse as 10-YEAR contracts.** 328 of 2,727
   distinct contracts carry `contract_years > 5`, which no NBA contract can;
   their $41k–$176k totals confirm 10-day money. 20 have `contract_start`
   inside 2019–2026. Excluded by rule 2 of the match. Written up as ISSUES #8
   because it also feeds `_filter_continuations`.

4. **Two rows sit above their ceiling by $50 and $70** — Bam Adebayo 2021 and
   Giannis 2021, at `cap_pct` 0.250000 and 0.350001. They disappear at any
   tolerance ≥ 1e-6. This is float dust at exactly the tier, matching the two
   such rows METHODOLOGY already describes; **METHODOLOGY's "over-cap count is
   zero" is correct** and no action is needed. Recorded only so the next person
   who runs a 1e-9 comparison does not re-open it.

5. **Unexplained, and I could not reduce it to one sentence.** Of 245
   start-anchored one-year candidate contracts, only 126 reconcile within 10%;
   the ratio `total / year-1 salary` has a clear second mass around **1.43**
   (75th pct 1.43, 95th 1.58). A one-year deal's total *is* its year-1 pay, so
   something is mismatched — plausibly Spotrac counting an option year while
   `contract_years` reads 1, or a cash-vs-cap-hit difference. **I did not test
   either hypothesis.** These rows are excluded by the reconciliation rule, so
   they cost coverage rather than correctness, and the exclusion is the
   conservative direction. Folded into ISSUES #9.

6. **Two instruments give different numbers for max-deal divergence** — nominal
   +16%, cap-weighted +0.3%. Per the escalation rule I want to be explicit that
   this is *not* an instrument disagreement: they measure different quantities,
   both correctly. Nominal AAV/signing-cap answers "what fraction of today's
   cap is the average yearly dollar figure", which is not a question anyone
   needs; the cap-weighted share answers "what fraction of the cap does this
   contract occupy on average", which is the one the target means.

---

## Files touched

| file | change |
|---|---|
| `docs/briefs/2026-07-23-aav-target-memo.RESULT.md` | new — this memo |
| `ISSUES.md` | added #8 (10-day parse bug) and #9 (AAV coverage) |

No pipeline, model, config or data file was modified. Nothing in another
agent's lane was touched.

**Branch**: `aav-target-memo`, cut from `c02263a`.

**A note on the pin.** The brief pins to `v7.8x`. I branched from `c02263a`
(HEAD) rather than the tag after verifying that `git diff v7.8x..HEAD` touches
no executable model code: the only non-documentation change is five `print`
statements inside `evaluate_suite.main()`. `load_evaluation_frame` returns the
same 1,172 rows with the same filter cascade (220/148/159/116/143/132/131/123
by season), so every number here reproduces from the tag. Branching from HEAD
was necessary because `docs/briefs/` does not exist at `v7.8x`.

**Reproduction.** The experiment scripts live in the session scratchpad, not
the repo, per the worker brief. The memo is self-contained: §2 states the
matching rule in full, and everything downstream is that rule plus
`load_evaluation_frame()` and `data/processed/spotrac_signing_types.csv`
(rebuilt 2026-07-23, 8,910 rows). The one non-obvious implementation detail is
that candidate contracts must be de-duplicated on
`(player_name_norm, contract_start, contract_years, total_value, aav)` before
matching, since the table stores one row per covered *season*.

---

## ISSUES.md additions

- **#8 — Spotrac 10-day contracts are parsed as 10-YEAR contracts.** 328 of
  2,727 contracts, 20 inside 2019–2026, with fabricated backward-walked seasons
  (Tolliver starting in 1980). Diagnostic table only, but it feeds
  `_filter_continuations`, whose span signal is exactly the shape a phantom
  10-year span produces. Screened out today only by the AAV-distance test, i.e.
  by luck rather than construction. Includes a reproduce command, the fix, and
  a verify step that requires the continuation filter still drop 119 rows with
  the same per-season split.
- **#9 — AAV is unavailable for half the evaluation frame, non-randomly.**
  Records the 51.5% coverage and its skew so the next agent proposing a
  contract-structure feature (length, guarantee share, option structure) knows
  up front that its missingness aligns with player quality and must be costed
  over a coverage-indicator arm, per gate 2.

---

## Proposed commit message

> Beyond the sections the brief requested; the architect edits and lands it. No
> version number is claimed — nothing moved a metric.

```
Reject the AAV target switch; the divergence is a denominator artifact

Measured against cap share rather than dollars, a five-year maximum
contract's average cap share is 1.0029 of its year-1 cap share (n=25,
realized caps only) — the CBA sets both the 5-8% escalator and the cap
path, and they cancel. The 16% nominal gap comes from dividing every
year of a deal by the signing-year cap.

Two further findings close the question. Both Stage-2 bounds lose the
property that makes them legitimate: an AAV ceiling needs contract
length and raise rate, neither knowable before signing, and at-floor
AAV ranges 0.52-1.62 of floor_pct so the floor is not a bound at all.
And on the 178 rows where AAV equals year-1 exactly, training on AAV is
worse (MAE +$0.21M, bias +$0.33M) — the model learns the multi-year
offset and applies it to short deals that do not have one.

Coverage is 51.5% and skewed (79% above $25M, 27% below $3M), so the
switch could only ever have been estimated on the top half of the
market. Filed as ISSUES #8 (10-day contracts parsed as 10-year) and #9
(AAV coverage skew).
```
