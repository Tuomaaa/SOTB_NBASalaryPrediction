# RESULT: the extension route — CBA raise caps, then the six-route mixture

**Brief**: `docs/briefs/2026-07-26-extension-route.md`
**Pin**: `master` (47d6da4), 944 rows, max zone 70
**Branch**: `worker/extension-route`

## What changed and why

ISSUES #21 is implemented. The CBA caps the first paying year of a veteran
extension at a multiple of the final-year salary of the contract being extended —
120% under the 2017 CBA, 140% under the 2023 CBA — **or the same multiple of the
league's published Estimated Average Player Salary, whichever is greater**.
`src/model/extension_cap.py` computes that ceiling; the multiplier and the EAS
figure are curated per season in
`data/raw/raw_external/extension_raise_caps.csv` with a source URL per row. Three
distinctions carry the whole result and each one is the fix for a way the
measurement failed before: the cap governs only the **first paying year**
(`span_start == season`, which drops 7 renegotiated seasons covered by an older
extension span); **rookie-scale extensions are capped by the tier, not by a
multiple**, classified by whether the previous season was a rookie-scale season
rather than by a text keyword; and **Designated Veteran extensions are exempt**,
tested by the supermax criteria at both the signing and the paying season. The
cap is attached as its own column and deliberately does not enter
`max_eligible_pct` or `is_max_contract` — the brief's adjudication that the raise
cap binds only conditional on choosing to extend.

The rule reproduces the data: **13 rows land on the legal multiple to the dollar
and 35 land on their computed cap to within $5,000**, with the 1.20/1.40 split
keyed on the SIGNING season showing zero crossings. The hard gate leaves **3
over-cap rows of 156**, each traced to a specific defect in our own salary table
(ISSUES #22), not to the rule. The honest headroom is **not** negligible: the
champion prices 11 rows above their corrected ceiling and snapping exactly those
is worth an oracle **ΔR² +0.0065**. Part 2 therefore ran, and its verdict is
split: the **full six-route mixture fails every gate** (ΔSel −0.0479, t = −2.50),
while the **extension route alone is the only positive ex-ante arm** — ΔSel
+0.0118, A2 +0.018, B1 **+0.0135**, C1 and C2 both improving — and fails only the
t > 2 bar at **t = 1.48**. Part 3's answer is negative and useful: the phase-3
collateral does not collapse, and the counterweight-band hypothesis in ISSUES #21
is refuted.

---

## Part 1 — the extension value function

### 1a. Classification

| | rookie-scale | veteran | total |
|---|---|---|---|
| first-paying-year extension rows | 73 | 83 | **156** |
| of which designated-veteran (exempt) | 0 | 14 | 14 |

By season: 2019 4/6, 2020 9/5, 2021 10/7, 2022 10/8, 2023 9/14, 2024 13/11,
2025 10/18, 2026 8/14 (rookie-scale / veteran).

**The extension flag.** A row is an extension row when its covering dated
contract has `is_extension = 1` **and** `span_start == season`. Of 163 rows whose
covering span is an extension, 7 fail the second test and are correctly excluded:
they are renegotiated seasons that the continuation filter keeps fresh under its
renegotiation carve-out while an older extension span still covers them. Myles
Turner 2022 is the worked example — his $35.07M is a renegotiated 2022-23 salary
sitting inside the span of his 2018 rookie extension, and applying a raise
multiple to it would have produced a ceiling $14M below his pay.

**The rookie-scale instrument.** `_load_rookie_scale_set` serves, used as "was
season s−1 a rookie-scale season". It needs no extension: every rookie-scale
extension inside the 2019-2026 frame starts at draft_year+4, so s−1 =
draft_year+3 is always inside the set's coverage. Against the `"rookie"` keyword
in `tx_text` — the instrument that failed the architect's second attempt — the two
agree on 151 of 156 and disagree on 5, and the disagreements are all keyword
errors: deals whose text omits the word for a player coming straight off a rookie
scale, and veteran extensions whose text mentions a rookie contract being
superseded.

**The designated-veteran carve-out** is the third distinction, and it is the one
the brief did not anticipate. A Designated Veteran (supermax) extension is
written at 30-35% of the cap and is outside the raise limit; Embiid 2023 is paid
exactly 35% of the cap at 1.42× his prior salary. Without the carve-out, 5
further rows fail the hard gate. Eligibility is tested at BOTH anchors because
each catches cases the other misses: the **signing** season catches deals signed
two summers before they pay (Wall and Harden both signed in 2017 for 2019-20
starts, by which season the debut-based service count has passed 9), and the
**paying** season catches players whose service years our debut count understates
(Embiid, two seasons on a roster before he debuted). The 14 rows are Lillard
2021, Booker 2024, Mitchell 2025, Giannis 2021, Harden 2019, Brown 2024, Tatum
2025, Embiid 2023, Wall 2019, Randle 2022, Towns 2024, Smart 2022, Jokic 2023,
Gobert 2021. **Smart 2022 is a false positive** and is filed as ISSUES #23.

### 1b. The knife edge, and why the multiplier keys on the SIGNING season

The ratio must be computed in dollars — `_load_prev_season_cap_pct` returns a
cap_pct, and the caps move 4-13% a year, so a cap_pct ratio blurs the knife edge
away entirely (it was the first thing this analysis got wrong).

| boundary | rows at exactly 1.200 | rows at exactly 1.400 | crossings |
|---|---|---|---|
| **signing season** (≤2022 / ≥2023) | 7 | 6 | **0** |
| paying season (≤2022 / ≥2023) | 6 + 1 in 2023 | 6 | 1 |

Signing-season keying is exact: Aaron Gordon 2022, Draymond Green 2020, Eric
Gordon 2020, Julius Randle 2022, Kevin Love 2019, Terry Rozier 2022 and Jimmy
Butler 2023 all sit on 1.200 and were all signed in 2022 or earlier; Derrick
White 2025, Brunson 2025, Jaren Jackson Jr. 2026, Jarrett Allen 2026, Josh Hart
2024 and P.J. Washington 2026 sit on 1.400 and were all signed in 2023 or later.
Keying on the paying season misplaces Jimmy Butler 2023 (signed 2021, paid at
exactly 1.200). This differs from `cba_era`'s boundary and both are right — see
ISSUES #24.

**The EAS route is real and is also keyed on the signing season.** Three rows
prove the keying, because their signing and starting years differ:

| row | signed | pays | paid | = published figure for |
|---|---|---|---|---|
| spencer dinwiddie 2019 | Dec 2018 | 2019-20 | $10,605,600 | 120% EAS **2018/19** ($10.61M) |
| daniel gafford 2023 | Oct 2021 | 2023-24 | $12,402,000 | 120% EAS **2021/22** ($12.4M) |
| naz reid 2023 | Jun 2023 (signing season 2022) | 2023-24 | $12,950,400 | 120% EAS **2022/23** ($12,950,400) |

Each lands on its signing season's figure, not its starting season's. Four more
land to the dollar on the same route: Zach Collins 2024 ($16,741,200 = 140% EAS
2023/24), and Nembhard 2025, Caruso 2025 and Wendell Carter Jr. 2026 (all
$18,102,000 = 140% EAS 2024/25). **35 of 83 veteran-extension rows sit within
$5,000 of their computed cap.**

### 1c. The curated table

`data/raw/raw_external/extension_raise_caps.csv`, whitelisted in `.gitignore`
next to `mle_exception_amounts.csv`. One row per signing season, with the CBA,
the multiple, the published cap figure, the implied EAS, a `published_precision`
column and a source URL.

| signing season | CBA | mult | published cap | precision | source |
|---|---|---|---|---|---|
| 2018 | 2017 | 1.20 | $10,610,000 | rounded to 10k | Hoops Rumors 2018-11 |
| 2019 | 2017 | 1.20 | $11,470,000 | rounded to 10k | Hoops Rumors 2019-12 |
| 2020 | 2017 | 1.20 | $12,000,000 | rounded to 100k | Hoops Rumors 2021-01 |
| 2021 | 2017 | 1.20 | $12,400,000 | rounded to 100k | Hoops Rumors 2021-10 |
| 2022 | 2017 | 1.20 | $12,950,400 | exact | Hoops Rumors 2022-11 |
| 2023 | 2023 | 1.40 | $16,741,200 | exact | Hoops Rumors 2023-11 |
| 2024 | 2023 | 1.40 | $18,102,000 | exact | Hoops Rumors 2024-11 |
| 2025 | 2023 | 1.40 | $19,418,000 | exact | Hoops Rumors 2025-11 |

Two disclosures. **The 2021 figure is published rounded to $12.4MM and Gafford
was paid $12,402,000** — $2,000 above the stored value, or 1.5e-5 in cap_pct,
inside the 1e-4 tolerance the gate uses. **Signing season 2017 is not in the
table**; the only row that needs it (Harden 2019) is designated-veteran exempt,
and the code prints a warning and falls back to the tier ceiling for any signing
season it lacks, so a missing figure can never produce a ceiling below pay.

### 1d. HARD GATE

**3 over-cap rows of 156** (tolerance 1e-4 cap_pct), down from 44 of 95 in the
architect's second attempt. All three fail in the same direction — the pay
implies a prior salary HIGHER than our table carries:

| row | pay | cap | over | our prior | prior the pay implies | gap |
|---|---|---|---|---|---|---|
| dejounte murray 2024 | $29.52M | $25.50M | $4.02M | $18.21M | $21.08M | +$2.87M |
| aaron gordon 2026 | $33.66M | $31.98M | $1.68M | $22.84M | $24.04M | +$1.20M |
| ivica zubac 2025 | $18.89M | $18.10M | $0.79M | $11.74M | $13.50M | +$1.75M |

Each is diagnosed, and none is a defect in the rule:

- **Murray** — his extension is 4yr/$114,238,204, which is exactly
  25.4996 / 27.540 / 29.580 / 31.620: a first year at exactly 140% of $18.214M
  with 8% raises. Our 2024 figure is $29,517,135; he was traded to New Orleans
  that July and a **trade bonus** is added to the remaining salary. The observed
  target is the negotiated price plus a bonus, which no ceiling rule can match.
- **Zubac and Gordon** — both are **renegotiate-and-extend** deals whose
  renegotiated prior-season salary our table does not carry. The arithmetic
  closes exactly on the renegotiated figure: 1.4 × $13,495,700 = $18,893,980 is
  Zubac's pay to the dollar, and 1.4 × $24,041,455 = $33,658,037 is Gordon's.
  Gordon's row carries the tell independently — our 2024 and 2025 both read
  $22,841,455, a duplicate, and his 4yr/$86.64M deal only sums if 2025 is
  $21.84M. Spotrac's text says "renegotiation-and-extend" for Turner 2022 and
  Sabonis 2023 but not for these two, so `renegotiated_seasons` cannot see them.

Filed as **ISSUES #22** with the repair path. I have NOT clamped the ceiling to
observed pay on these rows: the clamp would be target-reading and would hide
exactly the data defect the gate exists to surface. Their cost is bounded — the
extension cap is a Stage-2/told-parameter quantity that never touches
`max_eligible_pct` or the censor mask, so the three rows cost at most a
$4.0M/$1.7M/$0.8M underprediction each and cannot corrupt the target or
mislabel a row as censored.

### 1e. The honest headroom

The corrected ceiling is dramatically lower than the tier ceiling: **60 veteran
rows have their ceiling lowered by more than $0.1M, $1,080M in total, up to
$31.4M on one row** (Wendell Carter Jr. 2026, $49.49M → $18.10M; P.J. Washington
2026 $49.49M → $19.81M reproduces ISSUES #21's $29.7M headline).

The champion prices **11 rows above their corrected cap**. Snapping exactly those
to it:

| | champion | snapped | Δ |
|---|---|---|---|
| A1 CV R² | 0.7865 | **0.7930** | **+0.00650** |
| MAE | $3.076M | $3.008M | −$0.068M |
| bias | −$0.162M | −$0.241M | −$0.079M |

| row | pay | champion | cap | cut | MAE gain |
|---|---|---|---|---|---|
| ivica zubac 2025 | $18.89M | $39.26M | $18.10M | $21.16M | +$19.57M |
| derrick white 2025 | $28.10M | $41.99M | $28.10M | $13.89M | +$13.89M |
| jalen brunson 2025 | $34.94M | $45.17M | $34.94M | $10.22M | +$10.22M |
| eric bledsoe 2019 | $15.63M | $25.44M | $18.00M | $7.44M | +$7.44M |
| terry rozier 2022 | $21.49M | $26.69M | $21.49M | $5.20M | +$5.20M |
| dejounte murray 2024 | $29.52M | $30.32M | $25.50M | $4.82M | **−$3.22M** |
| jrue holiday 2021 | $30.13M | $34.94M | $31.05M | $3.89M | +$3.89M |
| wendell carter jr. 2026 | $18.10M | $20.90M | $18.10M | $2.79M | +$2.79M |
| toumani camara 2026 | $18.08M | $21.87M | $19.42M | $2.45M | +$2.45M |
| josh hart 2024 | $18.14M | $20.40M | $18.14M | $2.25M | +$2.25M |
| spencer dinwiddie 2019 | $10.61M | $10.71M | $10.61M | $0.10M | +$0.10M |

Total prediction cut $74.2M over 11 rows, mean MAE gain +$5.87M/row. Ten of the
eleven improve; the one that worsens is Murray, the trade-bonus row from the hard
gate. **This headroom is not negligible** — it is three times the +0.002 feature
bar — so Part 2 was run.

---

## Part 2 — the six-route mixture

### Class set and centering

`route_mixture.ROUTE6_CLASSES = [capspace, max, mle, floor, bird, extension]`,
priority max > floor > mle > extension > bird > capspace, so the max zone (70)
and floor zone are bit-identical to the four-class set and every phase-1..3
number stays comparable. `capspace` is every non-Bird continuous row — the
route-delta RESULT measured δ_capspace and δ_other as +0.00083 ± 0.00498 and
+0.00045 ± 0.00226, both indistinguishable from zero, which is the evidence for
merging them.

| route | n | OvR AUC | P̄ | P̄ on true | P̄ on false |
|---|---|---|---|---|---|
| capspace | 272 | 0.7235 | 0.298 | 0.423 | 0.248 |
| max | 70 | 0.9722 | 0.064 | 0.625 | 0.019 |
| mle | 93 | 0.7106 | 0.091 | 0.153 | 0.084 |
| floor | 240 | 0.8251 | 0.257 | 0.483 | 0.181 |
| bird | 167 | 0.7830 | 0.179 | 0.326 | 0.147 |
| **extension** | **102** | **0.9891** | 0.110 | **0.780** | **0.029** |

**The centering.** Route values are composed as

```
pred = f(x) + [ Σ_k P_k·V_k(x) − Σ_k π_k·V_k(x) ]
```

with π the training-slice route frequencies. Feed P = π and the bracket is
identically zero. **Verified numerically inside every (seed, fold):
max |mixture(P=π) − f(x)| = 5.55e-17.** Equivalently
`pred = f + Σ_structural (P_s − π_s)(V_s − f) + (P_bird − π_bird)δ_bird + …`, so
only a row whose route distribution deviates from the population average moves —
the property the route-delta failure showed was missing.

Raw values: `V_max = max_eligible_pct`, `V_floor = floor_pct`, `V_mle` = the
season's exception amount nearest f(x), `V_bird = f + δ_bird`,
`V_cap = f + δ_cap`, and **`V_ext = min(f + δ_ext, ext_value_pct)`** — the cap is
a bound, not a price, and most extensions sit below it, so applying it as a
ceiling on the route value rather than as the value itself is what makes the
route a sparse intervention. `ext_value_pct` is the ex-ante counterfactual
ceiling every row would face (an extension paying in s is signed in s−1), defined
for 857 of 944 rows; the 87 without an observable prior salary are left at f(x).

Per-fold intercepts (seed 0): δ_bird +0.0144/+0.0099/+0.0125/+0.0108/+0.0102,
δ_capspace +0.0001/−0.0009/+0.0011/+0.0003/+0.0008, δ_extension
+0.0133/+0.0105/+0.0171/+0.0074/+0.0147. π ≈ [0.29, 0.07, 0.10, 0.25, 0.18, 0.11].

### Numbers table

| arm | A1 | A2 | ΔSel | SE | t | slope | C1 excess | MAE | bias |
|---|---|---|---|---|---|---|---|---|---|
| champion | 0.7865 | 0.8341 | — | — | — | 0.9809 | — | $3.076M | −$0.162M |
| **full (six routes)** | 0.7436 | 0.8063 | **−0.04789** | 0.01914 | **−2.50** | 0.8194 | +0.1614 | $3.080M | −$0.057M |
| **ext_only** | **0.7984** | **0.8522** | **+0.01178** | 0.00797 | **+1.48** | 0.9840 | **−0.0032** | **$2.964M** | −$0.156M |
| struct_only | 0.7454 | 0.8068 | −0.04655 | 0.01828 | −2.55 | 0.8209 | +0.1599 | $3.069M | −$0.124M |
| delta_only | 0.7842 | 0.8320 | −0.00202 | 0.00133 | −1.52 | 0.9799 | +0.0010 | $3.103M | −$0.120M |
| full, λ=0.50 | 0.7787 | 0.8315 | −0.01110 | 0.00919 | −1.21 | 0.8804 | +0.1004 | $2.960M | −$0.104M |
| full, λ=0.25 | 0.7881 | 0.8379 | −0.00022 | 0.00467 | −0.05 | 0.9238 | +0.0571 | $2.968M | −$0.130M |
| ext_cap_only | 0.7915 | 0.8437 | +0.00482 | 0.00534 | +0.90 | 0.9894 | −0.0086 | $3.033M | −$0.194M |
| ext_delta_only | 0.7918 | 0.8399 | +0.00531 | 0.00174 | **+3.05** | 0.9738 | **+0.0070** | $3.024M | −$0.119M |
| *ex-post (context only)* | *0.8284* | *0.8737* | *+0.05541* | *0.01200* | *+4.62* | *0.8547* | *+0.1261* | *$2.579M* | *+$0.001M* |

Per-fold ΔSel:

- full: −0.0694, +0.0139, −0.0211, −0.0876, −0.0752
- **ext_only: +0.0090, +0.0060, +0.0418, +0.0079, −0.0059**
- struct_only: −0.0680, +0.0118, −0.0193, −0.0826, −0.0746
- delta_only: −0.0028, +0.0032, −0.0042, −0.0037, −0.0027
- ext_cap_only: +0.0034, +0.0018, +0.0253, −0.0008, −0.0056
- ext_delta_only: +0.0039, +0.0037, +0.0100, +0.0085, +0.0004

B1 forward (rolling-origin 2024-26): champion **0.8342**, full **0.7943**
(−0.0399), **ext_only 0.8477 (+0.0135)**. Per origin, ext_only: 2024 0.8807,
2025 0.8387, 2026 0.8111 (champion 0.8757 / 0.8173 / 0.7964) — it improves every
origin.

The λ arms are a diagnostic curve, reported in full. **No λ is selected on a
score**; picking one on ΔSel would be the same one-way valve the sigma sweeps
closed.

### Gate verdicts

| gate | bar | full | ext_only |
|---|---|---|---|
| 1. paired ΔSel t on the selection pool | > 2 | −2.50 **FAIL** | +1.48 **FAIL** |
| 2. A2 same direction as ΔSel | — | −0.0278 PASS | +0.0181 PASS |
| 3. C2 fixed-segment \|bias\| growth | ≤ $0.30M | +$0.48M **FAIL** | +$0.07M PASS |
| 4. B1 forward drop | ≤ 0.003 | +0.0399 **FAIL** | −0.0135 PASS |
| 5. C1 \|slope−1\| excess vs incumbent | ≤ 0.005 | +0.1614 **FAIL** | −0.0032 PASS |
| | | **0 of 5** | **4 of 5** |

### The extension route's two ingredients pull in opposite ways

`V_ext = min(f + δ_ext, ext_value_pct)` carries two separable effects. Splitting
them (`ext_mode` in the harness) is the most decision-relevant table here:

| arm | ΔSel | SE | t | per-fold sign | slope | C1 excess | C2 worst |
|---|---|---|---|---|---|---|---|
| ext_cap_only (Part 1's ceiling, no premium) | +0.00482 | 0.00534 | +0.90 | + + + − − | 0.9894 | **−0.0086** | +$0.14M |
| ext_delta_only (premium, no ceiling) | +0.00531 | 0.00174 | **+3.05** | + + + + + | 0.9738 | **+0.0070** | +$0.13M |
| ext_only (both — the ship form) | +0.01178 | 0.00797 | +1.48 | + + + + − | 0.9840 | −0.0032 | +$0.07M |

The two contribute almost equally in the mean (+0.0048 and +0.0053, summing to
the combined +0.0118) and have opposite statistical characters:

- **The cap is lumpy.** t = 0.90, with one fold carrying +0.0253 and two folds
  slightly negative. That is what a sparse legal ceiling on 11 rows looks like —
  it either catches a fold's big overprediction or it does not.
- **The premium is broad.** t = 3.05 with all five folds positive: extensions are
  systematically underpriced by the pooled surface (route bias −$1.62M) and
  δ_extension ≈ +0.011..+0.017 cap_pct corrects it everywhere. But it costs
  calibration — C1 excess +0.0070, **above the 0.005 bar** — for the same reason
  δ_bird did in the route-delta work: a positive intercept applied through a
  nonzero P bleeds onto rows that will not extend.
- **They are complementary on calibration.** The cap pulls the slope back
  (−0.0086 on its own), which is exactly what cancels the premium's damage: the
  combined arm's C1 excess is −0.0032, better than either the incumbent or
  ext_delta_only. So the ship form is not the sum of two independent effects; the
  ceiling is what makes the premium safe.

B1 forward was run for `full` and `ext_only` only; the two decomposition arms are
CV-and-integrity diagnostics, not candidates.

C2 detail, ext_only: worst movement is Early Bird −$1.71M → −$1.78M (+$0.07M),
then Cap Space +$0.05M and Sign & Trade +$0.04M. C2 detail, full: Early Bird
+$0.48M is the only breach, but the whole table moves by $1-2M per segment (Bird
Rights −$2.39M → −$0.45M, Sign & Trade −$4.61M → −$2.80M) — the surface is being
reshaped, not corrected.

### Where the damage in the full mixture comes from

`struct_only` (−0.0466, t = −2.55) reproduces `full` (−0.0479, t = −2.50) almost
exactly, and `delta_only` is ~0 (−0.0020). **The loss is entirely in the
max/floor/mle routes.** The mechanism is one line: `V_max = hi` and
`V_floor = floor_pct` are far from f(x) for *every* row, so the centering term
−π_max·(hi − f) applies a systematic pull to the ~93% of rows that are not max
candidates. P(max) averages 0.019 on non-max rows against π_max = 0.069, so the
classifier is sharper than the population prior and the redistribution is large
in both directions: calibration slope collapses 0.981 → 0.819. Per route, the
structural routes win big and everyone else pays — max MAE $4.30M → $1.91M,
floor $2.10M → $1.29M, against bird $3.55M → $4.38M, mle $2.88M → $3.60M,
extension $4.83M → $5.92M, capspace $2.74M → $2.92M.

`V_ext` does not have that shape. It equals f + δ_ext wherever the cap does not
bind, so `(P_ext − π_ext)(V_ext − f)` is ≈ the small δ term on most rows and only
becomes large where a legal ceiling actually sits below the model's price. That
is why one route out of six survives composition with an imperfect P.

### Where ext_only acts, and what it costs the rest

| movement | rows |
|---|---|
| \|Δ\| > $0.1M | 632 of 944 |
| \|Δ\| > $0.5M | 127 |
| \|Δ\| > $1.0M | 74 |
| \|Δ\| > $2.0M | 15 |

Total absolute movement $289.3M; **mean signed movement +$0.006M** — the
centering holds in aggregate, there is no surface lift.

| | n | champion MAE | ext_only MAE | Δ | better/worse |
|---|---|---|---|---|---|
| rows moved > $1M | 74 | $5.26M | $4.31M | **−$0.95M** | 50 / 24 |
| everything else | 870 | $2.890M | $2.849M | −$0.041M | — |

So it is not free — 24 of the 74 moved rows get worse — but the collateral on the
untouched 870 is negative, which is what the C1/C2/B1 passes are made of. The
biggest single moves are Zubac 2025 (−$12.4M), Derrick White 2025 (−$9.2M),
Rozier 2022 (−$4.4M) and Bledsoe 2019 (−$4.2M) downward, against Jaren Jackson
Jr. 2026 (+$2.4M), Gafford 2026 (+$2.3M) and Nembhard 2025 (+$2.2M) upward. The
upward moves are the δ_extension premium on rows whose cap does not bind, not the
cap — extensions are underpriced by the pooled surface (route bias −$1.62M) and
the route carries both effects.

### Anomalies

1. **ΔSel +0.0118 on ext_only is above the +0.01 re-verify line.** It decomposes
   into +0.0048 from the legal ceiling and +0.0053 from the route premium, and
   both mechanisms are ordinary. The ceiling removes 11 predictions that exceed a
   legal bound, three of them by $10-21M (Zubac $39.3M → $18.1M); removing three
   $10M+ errors from 944 rows is worth roughly that much R², and the effect is
   fold-concentrated (fold 2 +0.0418) which is also why t = 1.48. The premium is
   a −$1.62M route bias being corrected, broad and stable across all five folds.
   Nothing about either is fold-dishonest: `ext_value_pct` is built from
   `prev_cap_pct` and a curated CBA table, both available before the season, and
   both P and δ are estimated inside the training slice.
2. **The ex-post/ex-ante gap is smaller here than in the route-delta work.**
   ex-post +0.0554 (t = 4.62) against ex-ante +0.0118 for the full mixture's
   −0.0479 — P(extension) at AUC 0.9891 is the most identifiable route in the
   set, which is why its ex-ante arm keeps a third of its oracle value while the
   full mixture keeps none.
3. **The ex-post arm's calibration is also bad** (slope 0.855, C1 excess +0.126)
   even though its ΔSel is strongly positive. Told-route composition with
   `V_max = hi` overfits the max rows to their ceiling. Read the ex-post number
   as an upper bound on route information, not as a model.
4. **`ext_value_pct` is missing for 87 rows.** All are players with no observable
   prior-season pay (first contracts and prehistory-boundary rows). They are left
   at f(x), which is the neutral choice, but it means the extension route cannot
   act on a first-time extendee whose prior salary we never saw.

---

## Part 3 — the phase-3 re-run

`scripts/eval_route_mixture_p3.py` was re-run **unmodified**. It reproduces the
stored run exactly: champion A1 0.7868, A2 0.8340, true-max zone n=70 MAE $4.28M
bias −$4.25M, counterweight n=32 bias −$4.96M, `champion_repro_max_absdiff`
5.3228e-03 (identical to the stored value), and all eight gate cells identical to
`route_mixture_p3_eval.json` — e.g. base_tau* win +$0.24M / dSel t +1.62,
enriched_tau90 win +$1.16M / dSel t −2.04.

**This is the expected result and it answers the question.** The extension cap
deliberately does not enter `max_eligible_pct` (the brief's own trap: right-
censoring extension rows is forbidden), so nothing upstream of p3 moves and the
max branch cannot change. The substantive question — *would the extension route
have removed the collateral?* — is answered by annotating p3's collateral rows
with their extension status:

| collateral row (base_tau90) | pay | champion | pushed | damage | tier ceiling | extension cap | binds? |
|---|---|---|---|---|---|---|---|
| james harden 2022 | $33.00M | $38.48M | $46.89M | **$8.41M** | $52.64M | — (not an extension) | no |
| marcus smart 2022 | $17.46M | $28.72M | $35.17M | **$6.45M** | $43.28M | $43.28M (DVP-exempt) | no |
| tyler herro 2023 | $27.00M | $30.96M | $34.01M | $3.04M | $34.01M | $34.01M (rookie-scale) | no |
| jalen brunson 2025 | $34.94M | $45.20M | $46.39M | $1.19M | $46.39M | **$34.94M** | **yes** |
| jaylen brown 2020 | $23.44M | $27.29M | $27.29M | $0.00M | $27.29M | $27.29M (rookie-scale) | no |

**The brief's premise is wrong in the way that matters.** Four of the five are
extension rows, not three — but the raise cap **binds on exactly one of them**,
Brunson, carrying **$1.19M of the $19.09M total damage: 6%, not 82%**. Two are
rookie-scale extensions, whose cap IS the tier ceiling by construction, and one
(Smart) is exempted by the designated-veteran carve-out. The two largest damage
rows, Harden $8.41M and Smart $6.45M, are untouched by the raise cap. If Smart's
designated-veteran over-grant is corrected (ISSUES #23) his cap becomes $16.61M
and the extension-capped share rises to 40% — still not 82%, and it requires a
separate fix.

**The counterweight-band hypothesis is refuted.** ISSUES #21 argued that the
band's $4.87M underprediction "is not model error at all" but a population of
extension-capped rows, and that the brake which closed the route-mixture line "is
measuring a data bug". It is not:

| counterweight band (non-max, 70-90% of ceiling) | n | champion bias |
|---|---|---|
| all | 32 | −$4.93M |
| extension rows | 12 | **−$0.97M** |
| non-extension rows | 20 | **−$7.30M** |

The underprediction lives in the non-extension rows — Kuzma 2023 (−$15.9M),
Rozier 2019 (−$14.9M), Horford 2019 (−$14.8M), Hartenstein 2024 (−$14.6M),
Claxton 2024 (−$13.5M), Jaren Jackson Jr. 2025 (−$11.6M) — all genuine large
contracts the model underprices, none of them extensions. Of the 12 extension
rows in the band, 8 are rookie-scale (unconstrained) and only two sit at exactly
1.000 of their extension cap (Kevin Love 2019 and Brunson 2025), and Love's is a
$17.4M **under**prediction, which a ceiling cannot fix.

**Recommendation for the queue**: the provisional marker on the phase-3 closure
can be lifted. The brake is measuring model error, not a data bug.

---

## Files touched

Branch `worker/extension-route` off `master` (47d6da4), in worktree
`../BBall-worker-extension-route`.

| file | what |
|---|---|
| `src/model/extension_cap.py` | **new** — the raise cap, the ex-ante value, the gate |
| `data/raw/raw_external/extension_raise_caps.csv` | **new** — curated per-signing-season table with source URLs |
| `.gitignore` | whitelist the new curated table |
| `src/model/route_mixture.py` | **added** `ROUTE6_CLASSES`, `compute_route6_labels`, `train_route6_classifier`; nothing existing changed |
| `scripts/eval_extension_cap.py` | **new** — Part 1 evidence harness |
| `scripts/eval_route_mixture_p4.py` | **new** — Part 2 evidence harness |
| `ISSUES.md` | **added** #22, #23, #24 |
| `docs/briefs/2026-07-26-extension-route.RESULT.md` | this file |

Not touched: `train.py`, `evaluate_suite.py`, `_compute_max_eligible`,
`_compute_floor`, `max_eligible_pct`, `is_max_contract`, the Stage-1 censor mask,
`FEATURE_COLS`, and every document in the docs agent's lane. The champion
reproduces bit-identically inside both new harnesses
(`max|diff| = 0.00e+00` against `evaluation_suite.json`'s
`fold_r2_selection`).

Artifacts (gitignored): `outputs/models/extension_cap_eval.json`,
`extension_cap_oof.csv`, `route_mixture_p4_eval.json`, `route_mixture_p4_oof.csv`.

---

## Recommendation

**Adopt Part 1 as infrastructure; do not adopt either Part 2 mixture arm as the
champion; lift the provisional marker on phase 3.**

1. **Land `extension_cap.py` and the curated table.** They ship no number change
   — `ext_cap_pct` and `ext_value_pct` are new columns that no shipped code path
   reads — and they close ISSUES #21's measurement, which was the stated purpose
   of this brief. The audit is dollar-exact on 35 of 83 rows and the hard gate is
   3, all three diagnosed to ISSUES #22.
2. **Do not adopt the six-route mixture.** 0 of 5 gates, ΔSel −0.0479 at
   t = −2.50, calibration slope 0.819. The decomposition localizes the failure
   precisely: `struct_only` reproduces it and `delta_only` is null, so the fault
   is composing `V_max`/`V_floor` — values far from f(x) for every row — with a P
   that is sharper than the population prior. This is the same lesson as phase 2
   and the route-delta work, now measured on the full class set.
3. **The extension route alone is the one arm worth a second look, and it does
   not pass today.** 4 of 5 gates, and the two it passes most convincingly are
   the ones that catch overfitting — B1 forward **improves** by +0.0135 on all
   three origins and C1 calibration improves. It fails gate 1 at t = 1.48. The
   decomposition says why, and says the failure is not the same as a null result:
   the ceiling half is real but lumpy (t = 0.90 on 11 rows) and the premium half
   is broad and significant (t = 3.05) but breaks C1 on its own. Neither half is
   adoptable alone; the combination is calibration-clean but noisy. Three honest
   paths, in the order I would take them:
   - **Repair ISSUES #22 first, then re-judge.** Murray is the single row where
     the intervention makes things worse (−$3.22M), and Zubac's and Gordon's
     renegotiated priors are the same defect sitting on the two largest downward
     moves. Fixing all three should both raise the effect and cut its variance.
   - **Then consider judging it on its zone**, the precedent CLAUDE.md sets for
     Grabit's censored sides ("judge a targeted intervention where it acts"; the
     pooled rule "still governs changes that act on every row"). On the 74 rows
     it moves by more than $1M, MAE falls $5.26M → $4.31M (50 better / 24 worse)
     while the other 870 rows improve by $0.04M. I have deliberately **not**
     claimed that route — swapping the gate after seeing the number is the
     architect's call, not a worker's, and the brief set the pooled bar.
   - Do **not** ship `ext_delta_only` on its t = 3.05 alone. It fails C1 at
     +0.0070, and it is the same intercept-bleed that failed the route-delta
     experiment; the ceiling is what makes it safe.
4. **Lift the provisional marker on the phase-3 closure** (`docs/QUEUE.md`). The
   counterweight brake is measuring model error on 20 non-extension rows, not a
   data bug, and the extension cap binds on 6% of the collateral damage.

## Proposed commit message

```
Extension raise caps (ISSUES #21) and the six-route mixture

The CBA caps a veteran extension's first paying year at 120% (2017 CBA) or
140% (2023 CBA) of the final-year salary of the contract being extended, or
the same multiple of the published Estimated Average Player Salary,
whichever is greater. src/model/extension_cap.py implements it;
data/raw/raw_external/extension_raise_caps.csv curates the per-signing-season
figures with source URLs.

Three distinctions carry it: the cap governs only the first paying year
(span_start == season, dropping 7 renegotiated seasons); rookie-scale
extensions are capped by the tier and are classified by the rookie-scale set,
not a text keyword; and Designated Veteran extensions are exempt, tested at
both the signing and the paying season. The multiple keys on the SIGNING
season — 7 rows sit on exactly 1.200 and all were signed by 2022, 6 sit on
exactly 1.400 and all were signed from 2023, zero crossings. 35 of 83 veteran
rows land within $5k of their computed cap. Hard gate: 3 over-cap rows of
156, all three traced to stale renegotiated priors or a trade bonus in our
own salary table (ISSUES #22).

The cap does not enter max_eligible_pct or the Stage-1 censor mask: it binds
only conditional on choosing to extend.

Honest headroom: the champion prices 11 rows above their corrected ceiling;
snapping exactly those is worth an oracle +0.0065.

Six-route mixture (max/floor/mle/bird/capspace/extension), route values
centered so that P = population frequency reproduces f(x) exactly (verified
5.55e-17): the full mixture fails all five gates (dSel -0.0479, t -2.50,
slope 0.819). The decomposition localizes the fault to V_max/V_floor, which
sit far from f(x) on every row; the continuous-delta arm is null. The
extension route alone passes 4 of 5 (dSel +0.0118, t 1.48; B1 +0.0135, C1 and
C2 both improving) and is not adopted. Its two ingredients split cleanly:
the legal ceiling is +0.0048 at t 0.90 (lumpy, 11 rows) and improves
calibration by 0.0086; the route premium is +0.0053 at t 3.05 (all five folds)
but breaks C1 by +0.0070 on its own. The ceiling is what makes the premium
safe.

Phase-3 re-run is bit-identical, as designed. Its collateral does not
collapse: 4 of 5 collateral rows are extensions but the raise cap binds on
one, carrying 6% of the damage. The counterweight-band hypothesis is
refuted — the band's -$4.93M bias is -$0.97M on its 12 extension rows and
-$7.30M on its 20 non-extension rows.
```

## ISSUES.md additions

- **#22** — prior-season pay is stale for renegotiate-and-extend rows, and BBRef
  pay carries trade bonuses. The three hard-gate survivors, with the implied
  prior for each and a repair path that needs no new fetch.
- **#23** — Marcus Smart 2022 gets a designated-veteran ceiling from a DPOY that
  postdates his extension signing; the ISSUES #19 mirror. Includes the
  consequence: fixing it moves him from "exempt" to "over-cap" until #22 lands.
- **#24** — `cba_era` (season ≥ 2024) and the extension multiple (signing season
  ≥ 2023) use different CBA boundaries and both are right. Documentation debt
  filed so the next agent does not "align" them.

ISSUES #21 itself is not deleted — that is the architect's call at landing.
Its "82% of the collateral damage" claim is corrected above to 6%.
