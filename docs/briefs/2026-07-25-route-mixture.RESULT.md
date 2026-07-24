# RESULT — signing-route probability function + the max branch (phase 1)

**Brief**: `docs/briefs/2026-07-25-route-mixture.md`
**Pin**: `master` @ 44e8071 (v7.10x line, 944-row frame). Worktree branch
`worker/route-mixture`.
**Verdict**: machinery delivered and validated; **the MAX branch does NOT gate.**
It wins its target zone by a wide margin (true-max MAE −$1.59M, >3× the bar)
but fails both brakes and two of three guardrails. The failure is not the
classifier — it is excellent — but the composition: a continuous push weighted
by raw P(max) smears probability mass onto genuine non-max rows and inflates
their bias. Recommend **do not adopt in this form**; the classifier ships as
inert machinery (flag default off) for phase 2.

---

## 1. What changed and why

Built the phase-1 machinery for the unified route-mixture architecture: a
4-class probability function over signing routes (`max` / `floor` / `mle` /
`continuous`, defined by **where the salary landed**, never by Spotrac labels)
plus the MAX structural branch. Classes are disjoint on this frame (56 / 240 /
93 / 555, zero overlap; Spotrac `signing_cat` corroborates — 52/56 maxes are
Bird Rights, all 240 floors are Minimum, 50/93 mles are MLE-labeled). The
classifier is a modest-capacity XGBoost `multi:softprob` (depth 3, η 0.03, 400
rounds), fit fold-honest on the suite's GroupKFold-by-player folds, 10 seeds
averaged. **P is an OUTPUT composition weight applied to the Grabit latent; it
never joins `FEATURE_COLS`.** This is the opposite side of the ledger from the
ablation graveyard's `P(mechanism | x)` = −0.0073 (line 216, METHODOLOGY): that
was a fold-honest probability fed BACK INTO the regression as an input feature;
this is a probability applied to the regression's output. The two do not
conflict, and the graveyard entry does not veto this architecture.

The MAX branch (ship form, push-then-clip): `adj = latent + P(max)·(1.05·ceiling
− latent)`, then Stage 2 clips into `[floor_pct, max_eligible_pct]` unchanged.
High-P rows cross the ceiling and land on it; low-P rows are untouched; the
ambiguous middle moves partway. The 1.05 margin is a **fixed constant** — not
tuned on zone MAE (that would re-open the one-way valve the sigma sweeps
closed). Integration is behind `make_maxbranch_fitter(enabled=…)`, default
**off**: with the flag off the fitter reproduces the champion **bit-for-bit**
(max|diff| = 0.0, asserted in the harness), so nothing ships changed.

---

## 2. Part 1 — classifier diagnostics (10-seed fold-honest OOF, all classes)

| class | AUC | median P on true | n_true | note |
|-------|-----|------------------|--------|------|
| **P(max)** | **0.9650** | **0.5217** | 56 | beats every pre-repair baseline |
| P(mle) | 0.7158 | 0.1207 | 93 | weak — the phase-2 gate on the MLE branch |
| P(floor) | 0.8233 | 0.4617 | 240 | reported for phase 2 |

**P(max) vs the 2026-07-24 pre-repair baselines** (AUC 0.9595 / median 0.356 /
22 non-max above 0.5): AUC **0.9650** (↑), median on true maxes **0.5217** (↑,
materially above 0.356 — the repaired `prev_cap_pct` is doing real work), non-max
rows above P=0.5 **13** (↓ from 22). Calibration (binned by predicted P) tracks
the empirical max-rate well, mild overconfidence at the top:

```
P(max) bin      n    meanP   emp_rate
[0.0,0.1)     854    0.007    0.012
[0.1,0.3)      35    0.192    0.343
[0.3,0.5)      14    0.417    0.429
[0.5,0.7)      14    0.595    0.571
[0.7,0.9)      18    0.790    0.722
[0.9,1.0)       9    0.928    0.778
```

**P(mle) = 0.7158** is the number the MLE-recon brief made the MLE branch
conditional on (its RESULT bounded the oracle prize at +0.0143). It is weak, as
expected: team-cap membership — which decides whether a player is offered an
exception — is invisible to player features. **No MLE branch this phase**; this
AUC says a learned P(mle) branch would be building on sand and phase 2 should
treat it cautiously.

**Phase-2 diagnostic — true maxes the classifier still misses.** Median P on
true maxes is 0.5217 (up from 0.356), but **22 of 56** true maxes stay below
P=0.3, so push-then-clip barely touches them (their pinned share is ~0). They
split into two kinds:

- **Booker class** (young/extension maxes priced on reputation the impact
  metrics discount) — exactly the population the parallel feature batch targets:
  Booker 2019 (0.11), Klay 2019 (0.18), Jaylen Brown 2024 (0.19), Trae Young
  2026 (0.02), JJJ 2022/2026 (0.01), Bam 2026 (0.17), Mobley 2025 (0.26),
  Quickley 2024 (0.12), Murray 2025 (0.29).
- **Genuinely borderline** deals the market itself priced near the max line
  (VanVleet 2023 0.05, Ayton 2022 0.25, Siakam 2020 0.24, Harris 2019 0.03,
  D'Angelo Russell 2019 0.07) — closer to irreducible.

Full list in `outputs/models/route_mixture_oof.csv` (`p_max_oof`, `route_label`).

---

## 3. Part 2 — the max branch, three arms on fixed v7.10x rows

Champion reproduction is exact: `fold_r2_selection` max|diff| vs stored
reference = 0.00e+00, A1 = 0.7849. So every paired delta below is candidate-
minus-champion on identical (fold, seed) cells.

**Reference (champion) on the fixed rows**: true-max zone (n=56) MAE **$5.48M**;
counterweight band (non-max, cap_pct ∈ [0.70, 0.90) of ceiling, n=44) bias
**−$4.10M**, MAE $6.61M; 25%+ predicted band (n=49) bias **+$0.52M**;
A1 0.7849, A2 0.8338, B1 0.8276.

| arm | true-max MAE | ΔMAE | cw bias | Δcw | 25%+ bias | Δ25 | ΔSel ± SE | t | A1 |
|-----|-------------:|-----:|--------:|----:|----------:|----:|-----------|--:|---:|
| champion | 5.48 | — | −4.10 | — | +0.52 | — | — | — | 0.7849 |
| **push_clip** (ship) | **3.89** | **−1.59** | −2.09 | **+2.01** | +2.67 | **+2.15** | −0.0104 ± 0.0079 | **−1.32** | 0.7752 |
| mean (r1) | 4.21 | −1.27 | −2.41 | +1.69 | +2.28 | +1.77 | −0.0075 ± 0.0068 | −1.11 | 0.7782 |
| hard P>0.5 (r2) | 4.27 | −1.21 | −3.21 | +0.89 | +2.25 | +1.73 | −0.0067 ± 0.0052 | −1.30 | 0.7834 |

push_clip per-fold ΔSel: `[−0.0256, +0.0158, −0.0005, −0.0227, −0.0188]`
(4 of 5 folds negative).

Full metric table for the ship form: A1 0.7752 (Δ−0.0097), A2 0.8305
(Δ−0.0033, bias −0.12→+0.45M), B1 0.8105 (Δ−0.0171).

### Pinned-share curve (fixed 1.05 margin — reported, not optimized)

Of true maxes at each P(max) level, the share whose push_clip OOF prediction
lands within 1% of the ceiling:

```
P in [0.0,0.1)  n=10   pinned 0.000
P in [0.1,0.3)  n=12   pinned 0.083
P in [0.3,0.5)  n= 6   pinned 0.000
P in [0.5,0.7)  n= 8   pinned 0.500
P in [0.7,0.9)  n=13   pinned 0.923
P in [0.9,1.0)  n= 7   pinned 1.000
```

The valve pins confident maxes (P≥0.7 → ≥92% pinned) exactly as designed. The
1.05 margin is doing its job; the branch's problem is not under-pinning true
maxes, it is over-pushing non-maxes (below).

---

## 4. Gate verdicts

| # | gate | threshold | push_clip | verdict |
|---|------|-----------|-----------|---------|
| 1 | **Win**: true-max zone MAE | ≥ −$0.50M | **−$1.59M** | ✅ PASS (3.2×) |
| 2a | **Brake**: counterweight band signed-bias growth | ≤ +$0.30M | **+$2.01M** | ❌ FAIL |
| 2b | **Brake**: 25%+ predicted-band bias growth | ≤ +$0.30M | **+$2.15M** | ❌ FAIL |
| 3a | **Guard**: ΔSel t | > −2 | −1.32 | ✅ PASS |
| 3b | **Guard**: C2 fixed-segment \|bias\| growth | ≤ $0.30M | 4 segments breach | ❌ FAIL |
| 3c | **Guard**: B1 forward drop | ≤ 0.003 | **+0.0171** | ❌ FAIL (5.7×) |

**Overall: does not gate.** Passes the Win bar and the ΔSel-t guard; fails both
brakes and two of three guardrails.

### Brake/guardrail detail

**C2 fixed-segment bias growth** (push_clip − champion, same rows):

```
Bird Rights   n=274  −2.51 → −1.54  (Δ+0.97)  ✗
Cap Space     n= 76  −0.16 → +0.23  (Δ+0.39)  ✗
Non-Bird      n= 20  +2.09 → +2.45  (Δ+0.36)  ✗
Sign & Trade  n= 13  −4.62 → −3.72  (Δ+0.90)  ✗
Early Bird    n= 46  −1.69 → −1.44  (Δ+0.24)
MLE           n=123  +0.65 → +0.82  (Δ+0.17)
Minimum       n=242  +2.00 → +2.10  (Δ+0.10)
Unknown       n=131  +0.38 → +0.57  (Δ+0.19)
Other         n= 18  +0.79 → +0.86  (Δ+0.07)
```

The breaching segments (Bird Rights, Sign & Trade) are exactly where the true
maxes live, so some lift is expected — but the growth lands on the whole
segment, not the max rows alone.

**B1 forward, per origin** — the decisive gate, and its shape is the whole story:

```
origin  champion R2   push_clip R2    champ maxzone MAE   push maxzone MAE
2024      0.8634        0.8612            $4.24M              $1.76M
2025      0.8138        0.7527            $2.86M              $1.49M
2026      0.7938        0.8192           $13.76M             $11.36M
```

Forward, the branch **improves the max zone in all three origins** (2026's
$13.76M zone MAE is the Booker-class maxes the classifier misses — still down
$2.4M). Yet overall forward R² drops, almost entirely in **origin 2025**
(−0.061): the classifier trained on 2019-24 assigned moderate P to a handful of
high-dollar 2025 non-max rows and push-then-clip overshot them. R² weights those
big-dollar misses heavily; the many small MAE wins do not compensate. This is
the same MAE-vs-R² arithmetic tension METHODOLOGY flags (§ "where R² and MAE
disagree… read both") — here it disagrees because the intervention helps many
rows a little and hurts a few large ones a lot.

---

## 5. Anomalies

- **The direction is right; the precision is not.** push_clip lowers MAE almost
  everywhere it touches — true-max zone −$1.59M, counterweight band −$0.31M
  (6.61→6.30), 25%+ band −$0.21M (4.18→3.97), forward max zone in every origin.
  What fails is **signed bias and R²**, because the push is weighted by raw
  P(max) and the classifier carries real probability mass (P ∈ 0.1–0.5) on
  genuine non-max rows: 35 rows in [0.1,0.3), 14 in [0.3,0.5). Every one of them
  gets nudged toward a ceiling it will not reach. No result here is "too good";
  the −$1.59M win is real and independently visible in the forward zone MAEs, so
  it is not a leak.
- **No gate-passing result with an unexplained mechanism** — the escalation
  trigger does not fire. The one surprising number (forward max zone improving
  while forward R² drops) is fully explained by the origin-2025 non-max
  overshoot above.
- **Integration caveat (not a defect):** `src/model/predict.py` currently ships
  **plain XGBoost with no Stage-2 clip and no censoring** — the whole Grabit +
  route-mixture stack lives in the evaluation/champion layer, not in production
  inference. Wiring any adopted branch into `predict.py` is a separate
  architect-lane step. Flagged so the "inert flag, default off" claim is not
  misread as already-wired-to-production.

---

## 6. Phase-2 recommendation

**Do not adopt the max branch as specified.** Keep the classifier as delivered
machinery (flag off). The branch's failure is structural to the *continuous
raw-P push*, not to the classifier or the 1.05 margin:

1. **The lever is push precision, not more capacity or a tuned margin.** The
   brakes fail on non-max rows carrying moderate P. A margin change cannot fix
   that (and must not be tuned on zone MAE anyway). The natural next form is a
   **precision-gated push** — apply the push only where P(max) clears a high
   threshold (e.g. the ≥0.7 region that is already ≥92% pinned and almost purely
   true-max), leaving the ambiguous middle to Stage 1 untouched. That is a
   different knob from the fixed margin the brief froze, and it directly targets
   the collateral. It should be the first thing phase 2 tries.
2. **Re-run at landing with the feature-batch winners before re-judging.** The
   brief pins this to master and says the machinery is feature-agnostic. A
   sharper classifier (better features → probability mass concentrating toward
   0/1) would shrink the moderate-P smear that sinks the brakes — this is the
   single most likely way the *current* push form could start to gate, and it
   costs only a re-run of `scripts/eval_route_mixture.py` on the merged frame.
   But note it is not guaranteed: the feature batch raises P on Booker-class
   maxes (helping the Win), while the brakes fail on *non-max* rows whose P the
   feature batch may or may not lower.
3. **MLE branch: hold.** P(mle) AUC 0.7158 confirms the route is only weakly
   learnable from player features. Do not build the MLE branch until either the
   feature set or a team-cap-context signal lifts that AUC materially.
4. **Floor branch (phase 2 proper):** P(floor) AUC 0.8233 is workable; the floor
   is already left-censored in Stage 1, so a floor branch would be a *pull-down*
   mirror of this push and inherits the same precision caveat — gate it on the
   floor zone with the same brake discipline.

---

## 7. Files touched

New (committed on `worker/route-mixture`):
- `src/model/route_mixture.py` — classifier (labels, `train_route_classifier`,
  `oof_route_proba`), `grabit_latent` (champion latent exposed pre-clip),
  `make_maxbranch_fitter` (push_clip / mean / hard, `enabled` default **off**).
- `scripts/eval_route_mixture.py` — the evidence harness that reproduces every
  number in this RESULT (`OMP_NUM_THREADS=6 python scripts/eval_route_mixture.py`).

Artifacts (gitignored `outputs/models/`, for the architect's review):
- `route_mixture_eval.json` — all Part 1/2 metrics.
- `route_mixture_oof.csv` — per-row P(max), route label, champion + push_clip OOF.

Not touched: `train.py`, `evaluate_suite.py` (accept/reject rules — architect's
lane), the target/filter chain, sigma/gates/k_floor/censor population, and the
docs in the docs-agent lane. The frame is unchanged at 944 rows; A1/A2/B1 for
the champion reproduce the stored reference exactly.

---

## 8. Proposed commit message (architect edits and lands)

```
Route-mixture phase 1: signing-route classifier + inert max branch

Add the 4-class signing-route probability function (max/floor/mle/continuous,
defined by where the salary landed) and the MAX push-then-clip branch, both
behind a flag default OFF — the champion is bit-identical unless a caller opts
in. P(max) is an OUTPUT composition weight, never a feature.

Classifier (10-seed fold-honest OOF): P(max) AUC 0.9650, median P on true maxes
0.5217 (up from 0.356 pre-repair), 13 non-max above P=0.5 (from 22). P(mle) AUC
0.7158, P(floor) AUC 0.8233.

Max branch does NOT gate and is NOT enabled. It wins its zone (true-max MAE
$5.48M -> $3.89M, -$1.59M) but fails the brakes (counterweight-band bias
+$2.01M, 25%+ band +$2.15M, both vs the +$0.30M gate), C2 (4 segments), and B1
forward (-0.0171 vs the 0.003 gate) — a continuous raw-P push smears moderate
probability onto non-max rows. Machinery kept for phase 2 (precision-gated push
+ re-run on the feature-batch frame). Analysis-only; no version number.
```

(No version number claimed — the brief notes a gating max branch would start
the v8.0 line; it did not gate.)

---

## 9. ISSUES.md additions

None. No new defect was found. The `predict.py`-ships-plain-XGBoost observation
in §5 is a known design posture (the Value Board prices via impact metrics; see
commit 44e8071) and is recorded here as an integration caveat, not filed as a
bug.
