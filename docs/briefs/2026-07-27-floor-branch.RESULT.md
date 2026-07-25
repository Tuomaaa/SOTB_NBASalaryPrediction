# RESULT: the floor branch — measured, and do not adopt

**Worker session, 2026-07-27.** Pin: `master` @ `aef713b` (v7.13x frame, 944
rows). Branch `worker/floor-branch`. Harness `scripts/eval_floor_branch.py`,
one command, ~3 minutes at 10 seeds:

```bash
OMP_NUM_THREADS=6 python scripts/eval_floor_branch.py
```

---

## 1. What was done and what it found

Phase 2 declared the floor branch NO-GO on an argument. This session measured
it. The brief's headroom reproduces exactly — the floor zone is 240 rows, MAE
$2.10M, bias **+$2.04M**, and pulling every true at-floor row onto its floor
takes A1 **0.7865 → 0.8261 (+0.0396)**, MAE $3.076M → $2.575M. The champion's
`fold_r2_selection` matrix reproduces the stored suite to `max|diff| = 0.00e+00`,
so every delta below is paired on bit-identical folds and seeds.

The branch nevertheless does not convert, and the reason is not the threshold.
**P(floor) is anti-ranked against the error the branch exists to fix: Spearman
= −0.611 inside the zone.** The half of the floor zone the classifier is most
confident about carries **15.6%** of the zone's over-prediction; the top quarter
carries **2.5%**, where an error-ordered selector would carry 78.7%. The
mechanism is one sentence and it is structural: a fallen star taking a minimum
looks, on every performance feature the model has, *more* like a well-paid
player than the average non-floor row does — which is simultaneously why the
champion overprices him and why the classifier cannot flag him. Both ex-ante
arms fail at the pre-registered operating point; the told arm passes everything
because it reads the answer.

---

## 2. Numbers

Champion = the shipped two-stage Grabit. τ\* = 0.10, selected by the
pre-registered rule in §3. All deltas paired by fold on the selection pool.

| | A1 (pooled CV) | A2 (2024-26) | B1 (forward) | ΔSel ± SE | t | zone MAE win (sel) |
|---|---:|---:|---:|---:|---:|---:|
| **champion** | 0.7865 | 0.8341 | 0.8342 | — | — | — |
| **A** pull, ex ante | 0.7271 | 0.7780 | 0.7548 | **−0.08389** ± 0.03376 | **−2.48** | +$1.48M |
| **B** pull+margin, ex ante | 0.7914 | 0.8417 | 0.8380 | +0.00265 ± 0.00494 | +0.54 | +$0.51M |
| **T** told (Stage-3) | 0.8261 | 0.8510 | 0.8527 | +0.04597 ± 0.01833 | +2.51 | +$1.93M |

Per-fold ΔSel:

| arm | f1 | f2 | f3 | f4 | f5 |
|---|---:|---:|---:|---:|---:|
| A | +0.0078 | −0.1031 | −0.0794 | −0.0481 | −0.1967 |
| B | +0.0178 | +0.0021 | −0.0050 | +0.0085 | −0.0102 |
| T | +0.0532 | +0.0119 | +0.0111 | +0.0423 | +0.1113 |

**Convention, stated as the brief requires.** Arm T is the **told-parameter
(Stage-3)** mode: it pulls the rows that *are* at the floor, reading
`is_at_floor`, an outcome label. Per the 2026-07-26 decision it is reported in
the same column as the ex-ante arms — but it is not an ex-ante model, it is not
gated as a candidate, and §6 flags a limit of that convention that is specific
to this route. Arms A and B are ex ante and use only fold-honest P(floor).

**Control arms.** Three classifiers were fit per (seed, fold) so the primary is
not confounded with the enrichment or the class-set change:

| classifier | P(floor) AUC | median P on a true floor | max P |
|---|---:|---:|---:|
| 4-class, `FEATURE_COLS` (phase-3 reference 0.8259) | **0.8260** | 0.480 | 0.967 |
| 4-class, enriched | 0.8247 | 0.455 | 0.976 |
| **6-class, enriched — PRIMARY** | 0.8251 | 0.450 | 0.973 |

The phase-3 baseline reproduces (0.8260 vs 0.8259). Neither the 20 enrichment
columns nor the six-route split moves P(floor) — all three sit within 0.0013 of
each other, so nothing below is an artifact of the classifier choice. The
classifier is well calibrated over its range (empirical floor rate 0.040 in
[0,0.1), 0.413 in [0.3,0.5), 0.760 in [0.9,1.0)); it is honest, it simply has no
signal where the money is.

---

## 3. τ selection — the pre-registered rule, and what it returned

Fixed in the harness docstring before any arm was scored: grid 0.10→0.94 step
0.02 (extended *downward* from the max sweep's 0.30 because P(floor) comes from
a 0.83-AUC classifier, not a 0.98-AUC one); expected win = Σ over touched
true-floor rows of the champion's over-prediction; expected collateral = Σ over
touched non-floor rows of P × (champion − floor); **τ\* = argmax(win − coll)**;
margin fixed at 0.95.

The whole zone carries **$496.5M** of champion over-prediction. Abridged sweep
(full table in `outputs/models/floor_branch_eval.json`; `*` rows are below the
pre-registered grid, diagnostic only, never selectable):

| τ | n≥τ | true floors | collateral | purity | capture | E[win] | E[coll] | **OBJ** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00\* | 944 | 240 | 704 | 0.254 | 1.000 | 496.5 | 431.5 | +65.0 |
| 0.08\* | 614 | 228 | 386 | 0.371 | 0.814 | 404.0 | 326.3 | +77.7 |
| **0.10** | 567 | 225 | 342 | 0.397 | 0.741 | 367.9 | 300.3 | **+67.7** |
| 0.20 | 423 | 192 | 231 | 0.454 | 0.494 | 245.4 | 216.5 | +28.9 |
| 0.30 | 324 | 169 | 155 | 0.522 | 0.381 | 189.0 | 147.8 | +41.2 |
| 0.34 | 285 | 160 | 125 | 0.561 | 0.353 | 175.3 | 112.0 | +63.3 |
| 0.44 | 216 | 125 | 91 | 0.579 | 0.182 | 90.6 | 81.9 | +8.6 |
| 0.52 | 166 | 103 | 63 | 0.620 | 0.102 | 50.5 | 38.6 | +11.9 |
| 0.70 | 90 | 64 | 26 | 0.711 | 0.027 | 13.2 | 12.1 | +1.0 |
| 0.90 | 25 | 19 | 6 | 0.760 | 0.000 | 0.2 | 1.8 | −1.7 |

**τ\* = 0.10**, and two things about it matter more than the value.

First, **the objective never separates from zero.** Expected win and expected
collateral track each other at every threshold — 367.9 vs 300.3 at τ=0.10,
189.0 vs 147.8 at τ=0.30, 54.4 vs 48.8 at τ=0.50, 13.2 vs 12.1 at τ=0.70. The
objective is a small difference of two large, nearly equal quantities, so its
argmax is noise-dominated: 0.10 (+67.7), 0.34 (+63.3), 0.24 (+53.9) and 0.32
(+53.6) are not meaningfully distinguishable. This is the honest headroom the
brief asked for, computed before any arm ran, and it says the *net* headroom of
a thresholded pull-down is approximately zero everywhere. The max side's
objective had a genuine interior peak; this one does not.

Second, **τ\* sits on the grid's lower edge**, so the grid bound is doing the
selecting. The diagnostic rows show why: the objective is still positive at
τ=0.00 (+65.0) and peaks at τ=0.08 (+77.7), i.e. the rule as written wants to
touch every row in the frame. That is a rule reporting failure, not an operating
point. It changes no verdict — arm A is worse at lower τ and arm B's t never
reaches 2 anywhere — but it is recorded as an anomaly in §6.

Pre-registered sensitivities: argmax under Σ|champ err| = 0.10 (identical);
under the realizable win Σ(|champ−y| − |floor−y|) = 0.34; restricted to the max
side's grid (τ≥0.30) = 0.34. The gate battery is at τ\* = 0.10 as pre-registered;
the full arm sweep across all 43 thresholds is reported in §4 so the architect
can read every alternative operating point.

---

## 4. Gate verdicts

Bars from the brief: floor-zone MAE ≥ +$0.30M; no non-floor segment's |bias|
grows > $0.30M; C2 fixed segments ≤ $0.30M; B1 drop ≤ 0.003; ΔSel t > 2 for the
ex-ante arms. Zone MAE that *decides* is computed on `~is_confirmation` rows
only (ISSUES #20a — 37 of the 240 floor rows are confirmation rows, and the
champion's zone MAE differs across the split: $2.06M selection vs $2.32M
confirmation). C2 gates on |bias| growth (#20c); every segment comparison uses
fixed champion-defined row groups (#20b).

| gate | bar | **A** (ex ante) | **B** (ex ante) | **T** (told) |
|---|---|---|---|---|
| floor-zone MAE win, selection rows | ≥ +$0.30M | +$1.48M ✅ | +$0.51M ✅ | +$1.93M ✅ |
| — same, pooled / confirmation | reporting | +1.41 / +1.01 | +0.50 / +0.44 | +1.97 / +2.18 |
| non-floor brake, worst \|bias\| growth | ≤ $0.30M | **+$1.71M ❌** | **+$0.44M ❌** | +$0.00M ✅ |
| C2 fixed segments, worst \|bias\| growth | ≤ $0.30M | **+$1.82M ❌** | **+$0.43M ❌** | +$0.00M ✅ |
| B1 forward drop | ≤ 0.003 | **+0.0794 ❌** | −0.0038 ✅ | −0.0185 ✅ |
| paired ΔSel t | > 2 | **−2.48 ❌** | **+0.54 ❌** | (+2.51, not gated) |
| **verdict** | | **FAIL** | **FAIL** | PASS (told) |

Both ex-ante arms fail on the same two brakes plus ΔSel. The failing segments
are not marginal and they are the ones the brief predicted: arm A moves Cap
Space bias −$0.29M → −$2.11M, Bird Rights −$2.39M → −$3.81M, MLE +$0.58M →
−$1.91M. Arm B is a sixth the size but breaches the same two (Cap Space +$0.43M,
`nonfloor_all` +$0.44M) while buying a ΔSel indistinguishable from zero.

**Arm B never passes at any τ.** From the full sweep, its best paired t is
**+1.75 at τ=0.34** (ΔSel +0.00308, zone win +$0.32M — barely over the bar), and
it is negative for every τ ≥ 0.52. Arm A's t is negative at every τ below 0.40
and indistinguishable from zero above it. There is no operating point on this
grid where a thresholded pull-down earns adoption, so the failure is not an
unlucky τ.

---

## 5. The named collateral at τ\* — a pull-down has no legal brake

342 non-floor rows are touched. The brief's warning is confirmed: this is a
$10M-class error, the same magnitude as the max side's, and nothing structural
stops it.

**Arm A — total damage +$496.96M** (worst rows):

| row | P(floor) | pay | mechanism | champ err | arm err | damage |
|---|---:|---:|---|---:|---:|---:|
| walker kessler 2026 | 0.180 | $30.23M | Bird Rights | −7.72 | −27.78 | **+20.06** |
| brook lopez 2023 | 0.157 | $25.00M | Bird Rights | −3.81 | −22.98 | +19.17 |
| miles bridges 2024 | 0.109 | $27.17M | Bird Rights | −1.63 | −20.29 | +18.66 |
| coby white 2026 | 0.134 | $22.84M | Bird Rights | −2.97 | −20.39 | +17.42 |
| bradley beal 2025 | 0.277 | $19.38M | MLE | −3.03 | −17.09 | +14.05 |
| patrick williams 2024 | 0.164 | $18.00M | Bird Rights | −1.97 | −15.84 | +13.86 |
| harrison barnes 2019 | 0.195 | $24.15M | Bird Rights | −8.93 | −22.26 | +13.33 |
| jordan clarkson 2023 | 0.180 | $23.49M | Bird Rights | −8.47 | −21.38 | +12.91 |

**Arm B — total damage +$19.00M**, worst rows Beal 2025 (+$3.92M), Kessler 2026
(+$3.62M), Sexton 2022 (+$3.37M), Dort 2022 (+$3.24M), Lopez 2023 (+$3.02M).
The margin form is 26× cheaper because the pull is P-weighted and P is small on
exactly these rows — but the same smallness is why it recovers so little.

Note what the damage rows have in common: **every one is a genuinely well-paid
player whose P(floor) is 0.11–0.28.** The classifier is not confidently wrong
about them; it is unconfident about everything, and a low threshold sweeps them
in. That is the direct consequence of a 0.83 AUC applied to a decision that
needs precision at the top.

For contrast, the touched *true* floors at τ\*: 225 of 240, and arm A does erase
their error (Oladipo 2021 +$24.92M → +$4.92M, Trent 2024 +$9.21M → $0.00M,
Griffin 2021 +$8.31M → $0.00M). The branch works on the rows it reaches. It just
cannot reach them without also reaching the other 342.

---

## 6. Part 3 — can the classifier see the fallen stars? No, and here is the number

| row | champion error | **P(floor)** | P percentile | P base4 | P enr4 |
|---|---:|---:|---:|---:|---:|
| victor oladipo 2021 | +$24.92M | 0.115 | 0.434 | 0.088 | 0.159 |
| kelly oubre jr. 2023 | +$18.13M | 0.082 | 0.354 | 0.051 | 0.069 |
| montrezl harrell 2022 | +$17.20M | **0.015** | 0.136 | 0.015 | 0.021 |
| andre drummond 2021 | +$16.36M | 0.049 | 0.266 | 0.033 | 0.105 |
| hassan whiteside 2020 | +$13.58M | **0.018** | 0.151 | 0.012 | 0.009 |
| reggie jackson 2020 | +$9.95M | 0.081 | 0.352 | 0.059 | 0.071 |
| chris paul 2025 | +$9.67M | 0.039 | 0.242 | 0.026 | 0.032 |
| marc gasol 2020 | +$9.37M | 0.036 | 0.229 | 0.017 | 0.029 |
| blake griffin 2021 | +$8.31M | 0.469 | 0.792 | 0.266 | 0.493 |

**Eight of the nine sit below the non-floor median P of 0.098.** The worst 12
floor rows carry 30.8% of the zone's error and their median P(floor) is 0.081,
against 0.450 for the zone as a whole. Only Griffin is visible, and only just.

The answer to the brief's question — threshold or data source — is **neither
exactly: it is the features, and the features are already in the model.**

| feature | worst 12 floor rows | rest of the zone | non-floor rows |
|---|---:|---:|---:|
| mpg | **28.21** | 16.80 | 23.65 |
| darko_dpm_z | **+0.310** | −0.557 | +0.154 |
| lebron_z | **+0.468** | −0.408 | +0.129 |
| rapm_z | **+0.337** | −0.350 | +0.151 |
| usage_pct | **21.00** | 16.47 | 18.77 |
| prev_cap_pct | **0.133** | 0.038 | 0.072 |
| award_score_cum | **0.896** | 0.043 | 0.661 |
| age | 29.75 | 28.67 | 26.36 |
| availability_3yr | 0.616 | 0.540 | 0.672 |

On minutes, on all three impact metrics, on usage, on prior pay and on award
history, the twelve worst floor rows score **above the average non-floor row** —
not merely above the rest of the zone. The only feature pointing the right way
is age (+1.1 years over the zone, +3.4 over non-floor), and `availability_3yr`
— the one feature that ought to encode "he got hurt" — reads 0.616, *higher*
than the rest of the floor zone's 0.540 and only 0.056 below the non-floor mean,
because these players were healthy and productive in the season being priced and
signed for the minimum in the summer after.

So this is not a case of a missing column the classifier would have used. The
information that Oladipo would take $2.4M in July 2021 is not in his 2020-21
box score, his impact metrics, his prior salary or his award history, because it
was not a fact about his production — it was a fact about a torn quad tendon in
May and a thin market in August. Availability, injury history and age×decline
interactions cannot recover it: two of the three are already present and point
the wrong way. Recovering it needs an instrument dated *after* the season being
priced — an offseason injury/transaction feed, or free-agency market state — and
that is a data-acquisition question, not a modelling one.

**What a better classifier is worth, quantified:** an error-ordered selector
touching the top 25% of the zone would capture 78.7% of its over-prediction; the
current P captures 2.5%. The gap between those two numbers is the entire prize,
and it is a ranking problem, not a threshold problem.

---

## 7. `floor_pct` imprecision (ISSUES #6) — it costs essentially nothing

`floor_pct` is a recovered quantity, so the brief asks what the recovery costs
on the touched rows. On at-floor rows it lands within **MAE $0.130M** of observed
pay (bias −$0.034M, p90 $0.420M). Priced end-to-end: pulling the zone onto
`floor_pct` gives A1 **0.8261**; pulling it onto *actual pay* gives **0.8263**.
**The recovered scale costs 0.0002 of A1 and $0.033M of pooled MAE** — 0.5% of
the +0.0396 headroom. A published minimum scale would be exact but would buy
nothing measurable here. `floor_pct` is not what is standing between this branch
and adoption.

---

## 8. Anomalies

**(a) Arm T's ΔSel is +0.046 — far above the +0.01 red-flag line — and its
source is fully explained.** T pulls the rows that are at the floor because it
is told which they are; it is arithmetically the oracle of arms A and B, and its
A1 of 0.8261 is exactly the brief's oracle number. It is not a model, and it is
reported only because the 2026-07-26 convention places told-mode numbers in the
same column.

**(b) A limit of the told convention that is specific to this route, for the
architect to rule on.** The convention's justification is that the route is
available at prediction time for both deployed uses. That holds for the floor
route in the *Contract Surplus* use — a signed minimum deal's route is known —
but it does **not** hold for the unsigned-free-agent use. The extension route
has the property that "an unsigned free agent is by definition not extending";
the floor route has no equivalent, because "will he sign for the minimum" is
precisely the quantity a valuation of an unsigned free agent is producing.
Arm T's +0.046 is therefore a legitimate told-mode number for signed contracts
and is **not available at all** for the Value Board's unsigned-FA view. This is
a convention question, not a measurement one, so it is flagged and not decided.

**(c) The pre-registered objective's collateral term does not model arm A, and
one τ is applied to both arms.** Measured against what the arms actually did at
τ\*:

| | expected | realized zone reduction | realized non-floor damage | realized net |
|---|---:|---:|---:|---:|
| A | win 367.9 / coll 300.3 | $338.1M | **$514.9M** | **−$176.8M** |
| B | win 367.9 / coll 300.3 | $118.8M | $19.7M | +$99.1M |
| T | — | $473.3M | $0.0M | +$473.3M |

The win term is accurate for arm A (367.9 expected vs 338.1 realized, 1.09×) but
the collateral term understates arm A's actual damage by **1.7×**, because
`P × (champ − floor)` is the expected damage of a *P-weighted* pull — arm B's
form — while arm A pulls the whole way regardless of P. The objective's sign is
consequently wrong for arm A: it predicted +$67.7M net where arm A delivered
−$176.8M. The max side never hit this because its adopted arm was the P-weighted
push, so its collateral term happened to match. Filed as **ISSUES #25**; it
changes nothing here (arm A fails every gate anyway) but it would mis-select for
a future unconditional branch.

**(d) τ\* on the grid's lower edge.** Covered in §3. The pre-registered lower
bound of 0.10 is load-bearing for the *value* of τ\* but not for any verdict; the
diagnostic rows below it are printed so the shape is auditable rather than
implied.

**(e) Not an anomaly, recorded so it is not rediscovered:** arm A's realized
floor-zone win ($338.1M) is smaller than the oracle's ($473.3M) even though it
touches 225 of 240 true floors, because 15 of the highest-error rows sit below
τ and because arm A also un-fixes rows the champion had right — Oladipo 2021
lands at +$4.92M rather than $0.00M because the seed-averaged OOF blends folds
where P < τ with folds where P ≥ τ.

---

## 9. Recommendation

**DO NOT ADOPT** either ex-ante arm. Arm A fails five of five gates and would
cost 0.059 of A1 and 0.079 of forward R². Arm B fails three of five, buys a ΔSel
of +0.0027 (t = 0.54) that is indistinguishable from zero at its selected point
and never exceeds t = 1.75 anywhere on the grid, and breaches the Cap Space and
non-floor bias brakes to get it.

**Do not re-open this as a threshold problem.** The measured obstruction is the
ranking: Spearman(P, over-prediction) = −0.611 inside the zone, top-quartile
capture 2.5% against an achievable 78.7%. Every threshold rule inherits that
ordering, so no operating point, margin, or purity target changes the answer.
The three classifier variants agree to within 0.0013 of AUC, so classifier
tuning will not change it either.

**The headroom is real and stays on the board — +0.0396 of A1, six times the max
side's — but it is now priced as a data-acquisition question, not a modelling
one.** The rows that carry it are ones whose minimum salary was determined by
information dated after the season being priced. If the architect wants this
headroom, the next dispatch is an instrument that sees the offseason: transaction
or injury-report data between season S−1 and the signing, or free-agency market
state at signing. That is a scraping brief with a coverage-missingness gate
(ISSUES #12's channel), not another branch brief.

**Two things are worth landing from this session regardless of the verdict:**
the harness itself, which is the first working measurement of the floor side and
reproduces the champion exactly; and ISSUES #25, which is a live defect in the
τ-selection rule that the max side got away with by accident.

---

## 10. Files touched

| file | status |
|---|---|
| `scripts/eval_floor_branch.py` | **new** — the evidence harness (one command, 10 seeds, ~3 min) |
| `docs/briefs/2026-07-27-floor-branch.RESULT.md` | **new** — this document |
| `ISSUES.md` | **modified** — added #25 (τ-objective collateral term) |
| `outputs/models/floor_branch_eval.json` | generated (gitignored) — every number above |
| `outputs/models/floor_branch_oof.csv` | generated (gitignored) — per-row P, champion and all three arms |

Nothing in `src/` was touched: this is an output-side experiment and the
Stage-1 left-censoring (`floor_gate_k`, `sigma_left`) was not modified, per the
brief. `outputs/models/evaluation_suite.json` and `oof_reference.csv` were read
only — the harness writes under its own filenames and never triggers the suite's
`_prev` rotation.

Branch: **`worker/floor-branch`** (worktree `../BBall-worker-floor`, from
`master` @ `aef713b`). The harness placement follows the precedent of
`eval_route_mixture_p3.py` / `eval_max_branch_tau_sweep.py` — committed rather
than left in a scratchpad — because the RESULT's numbers and ISSUES #25's
reproduction command both point at it.

---

## 11. Proposed commit message

The architect edits this and lands it; no version number is claimed, and none is
warranted — no shipped number moves.

```
Floor branch measured and rejected: the classifier is anti-ranked against the error

Phase 2's NO-GO was an argument; this is the measurement. The headroom
reproduces (floor zone n=240, MAE $2.10M, bias +$2.04M; oracle A1
0.7865 -> 0.8261, +0.0396) and both ex-ante arms still fail.

At the pre-registered tau* = 0.10 (argmax of expected win minus expected
collateral, grid 0.10-0.94):

  A  pull            A1 0.7271  dSel -0.08389 (t -2.48)  5 of 5 gates fail
  B  pull + margin   A1 0.7914  dSel +0.00265 (t +0.54)  3 of 5 gates fail
  T  told (Stage-3)  A1 0.8261  dSel +0.04597 (t +2.51)  passes; reads the label

Arm B never reaches t > 2 at any threshold (best +1.75 at tau=0.34), so
the failure is not an unlucky operating point. The obstruction is the
ranking: Spearman(P(floor), champion over-prediction) = -0.611 inside the
zone; the top quartile by P carries 2.5% of the zone's over-prediction
where an error-ordered selector would carry 78.7%.

The mechanism is structural. On mpg, all three impact metrics, usage,
prev_cap_pct and award history, the twelve worst floor rows score ABOVE
the average NON-floor row -- which is why the champion overprices them and
why no classifier fed these features can flag them. availability_3yr reads
0.616 for them, higher than the rest of the zone: they were healthy in the
season being priced and got hurt afterwards. The headroom is real and needs
an instrument dated after the season, not a better threshold.

Also measured: the objective never separates from zero (expected win and
collateral track each other at every tau), floor_pct's recovered scale
costs only 0.0002 of A1 against a published one, and the collateral term
mis-models an unconditional arm by 1.7x (ISSUES #25).

Adds scripts/eval_floor_branch.py and the RESULT; no src/ change, no
shipped number moves.
```

---

## 12. ISSUES.md additions

One entry added — see `ISSUES.md` **#25**. Nothing else found in this session
was both real and unfixed: the brief's traps were all confirmed rather than
violated, `floor_pct`'s imprecision is quantified in §7 and is not worth a
maintenance entry, and the Stage-1 censoring was not touched.
