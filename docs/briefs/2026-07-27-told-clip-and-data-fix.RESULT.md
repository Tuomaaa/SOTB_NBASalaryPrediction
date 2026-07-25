# RESULT — ISSUES #22/#23 fixed, told-route Stage-3 clip re-measured

**Branch**: `worker/told-clip`, pinned to `aef713b` (master at dispatch).
**Harness**: `OMP_NUM_THREADS=6 python scripts/eval_told_clip.py` — one fit pass,
10 seeds x 5 GroupKFold folds, everything else a vector operation on the stored
predictions.
**Artifacts**: `outputs/models/told_clip_eval.json`, `told_clip_oof.csv`.

> **Convention note, and it is load-bearing.** Every number in the candidate
> column below uses the **told-route** convention adopted by the user on
> 2026-07-26: the signing route is an available covariate at prediction time and
> the model is scored with it, in the same column as the ex-ante numbers.
> **v7.1x–v7.13x were computed under the old "ignore the route" convention and
> are not naively comparable to the candidate arm here.** The champion column is
> computed under both conventions identically (it ignores the route), so the
> champion-vs-candidate *delta* is honest; the candidate's absolute A1 is not
> comparable to a published v7.13x figure.

## 1. What changed and why

Two data defects were repaired, and neither touches the model. **ISSUES #23**:
`extension_cap._designated_veteran` granted Marcus Smart a Designated-Veteran
exemption for 2022 off the DPOY he won in 2021-22 — nine months *after* he signed
the extension in August 2021 — because the award test was unioned over the signing
and the paying season. The award anchor is now the **signing season alone**
(eligibility is judged when the deal is signed; awards land in the spring, before
a July-June league year opens), while the service-year anchor stays unioned,
because what is unreliable there is our debut-based service count, not the
calendar. Exactly one row flips. **ISSUES #22**: the three over-cap rows are
repaired through a new curated table,
`data/raw/raw_external/salary_corrections.csv`, with two kinds of entry —
`prior_base` (the season's contractual base salary, applied inside
`_load_prev_season_cap_pct` so every consumer sees one number) and
`pay_above_base` (money inside observed pay that the raise cap does not govern: a
trade bonus, an earned incentive, added on top of whichever ceiling binds). No
ceiling is clamped to observed pay by rule; each entry carries its derivation and
a `confidence` field, and two of the five are *not* `verified` — see Anomalies.

The repaired data was then re-measured on the arm exactly as dispatched, at the
pre-registered **tau = 0.52**, MARGIN 1.05, not re-tuned. The frame is unchanged
in every respect that could make the comparison dishonest: **944 rows, targets
identical to 1e-16, `max_eligible_pct` and `tier_ceiling_pct` identical, route
labels identical**. The fixes move **four ceiling values and nothing else**, so
the champion, both route classifiers and the push are bit-identical before and
after, and the pre-fix arm below reproduces the architect's dispatch table to four
decimals.

## 2. Numbers

All arms scored on the same 944 rows, same folds, same 10 seeds. `dSel` is the
paired fold delta against the champion over selection-pool rows only.

| arm | A1 | A2 | B1 fwd | MAE | max-zone MAE | cw bias | 25%+ bias | dSel | SE | t | C2 worst |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | 0.7865 | 0.8341 | 0.8342 | $3.076M | $4.297M | −4.93 | +1.06 | — | — | — | — |
| push only | 0.7849 | 0.8381 | 0.8174 | $3.026M | **$3.179M** | −4.85 | **+2.27** | −0.00573 | 0.0035 | −1.63 | +0.03 |
| push + told clip **[PRE-FIX data]** | 0.7936 | 0.8514 | — | $2.945M | $3.179M | −5.33 | +1.13 | +0.00375 | 0.0087 | +0.43 | +0.02 |
| push + told clip **[#23 only]** | 0.7987 | 0.8514 | — | $2.921M | $3.179M | −5.33 | +1.13 | +0.00824 | 0.0078 | +1.06 | +0.02 |
| push + told clip **[#22 only]** | 0.7937 | 0.8517 | — | $2.939M | $3.179M | −5.33 | +1.15 | +0.00407 | 0.0086 | +0.47 | +0.02 |
| **push + told clip [repaired]** | **0.7988** | **0.8517** | **0.8355** | **$2.916M** | **$3.179M** | −5.33 | +1.15 | **+0.00856** | 0.0077 | **+1.11** | +0.02 |
| push + ex-ante clip (P(ext) ≥ 0.50) | 0.7903 | 0.8490 | 0.8204 | $2.975M | $3.179M | −5.12 | +1.64 | +0.00032 | 0.0089 | +0.04 | +0.01 |
| told clip, **no push** | 0.7945 | 0.8472 | — | $2.990M | $4.297M | −5.37 | +0.14 | +0.00881 | 0.0060 | +1.48 | +0.29 |

The dispatch table is reproduced exactly by the PRE-FIX row (0.7936 / $2.945M /
$3.179M / +1.13, dSel +0.0037 at t = 0.43).

**Per-fold dSel** (the fold is the unit; fold sd swamps seed sd):

| arm | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| push only | −0.0129 | −0.0114 | +0.0045 | −0.0096 | +0.0008 |
| push + told clip [PRE-FIX] | −0.0115 | +0.0029 | +0.0367 | −0.0091 | −0.0002 |
| **push + told clip [repaired]** | **+0.0110** | +0.0029 | +0.0367 | −0.0091 | +0.0014 |
| push + ex-ante clip | −0.0126 | −0.0114 | +0.0350 | −0.0091 | −0.0003 |
| told clip, no push | +0.0071 | +0.0037 | +0.0322 | +0.0005 | +0.0007 |

The repair turns fold 1 from −0.0115 to +0.0110 (that is Smart, who lives there)
and leaves the sign pattern otherwise unchanged. Four of five folds are now
positive, but fold 3 still carries most of the mean — hence t = 1.11.

### Control arms run

- **push only** — isolates the clip from the push. The push alone *loses* on the
  pooled statistic (dSel −0.0057) and fails the 25%+ band brake at +2.27; the clip
  is what makes the branch viable, which is the dispatch's finding, re-confirmed.
- **ex-ante clip** — the same arm clipping by `P(extension) >= 0.50` at the
  counterfactual `ext_value_pct` instead of the realized route. It keeps only
  +0.0003 of the +0.0086: the classifier does not find these rows. This is the
  honest measure of what the branch is worth **without** told information, and it
  is reported here so the two are never confused.
- **told clip, no push** — the clip on its own. Nearly the same dSel (+0.0088) at
  a higher t (1.48), and it *improves* the 25%+ band bias (+1.06 → +0.14), but it
  wins none of the max-zone MAE (stays at the champion's $4.297M) and its C2 worst
  segment is +0.29, one cent inside the bar. Offered as a datum, not a proposal.
- **#23-only / #22-only** — the decomposition below.

### Per-row, before and after

$M. `ceilPRE/ceilNEW` are the row's own extension ceiling before/after the fixes.

| row | pay | champion | push | ceilPRE | ceilNEW | clipPRE | clipNEW | errPRE | errNEW |
|---|---|---|---|---|---|---|---|---|---|
| marcus smart 2022 | 17.46 | 28.60 | 39.68 | 43.28 | **17.46** | 39.68 | 17.46 | **+22.22** | **0.00** |
| lamarcus aldridge 2019 | 26.00 | 27.79 | 38.20 | 26.82 | 26.82 | 26.82 | 26.82 | +0.82 | +0.82 |
| jalen brunson 2025 | 34.94 | 45.17 | 46.39 | 34.94 | 34.94 | 34.94 | 34.94 | 0.00 | 0.00 |
| dejounte murray 2024 | 29.52 | 30.32 | 30.32 | 25.50 | **29.52** | 25.50 | 29.52 | −4.02 | **0.00** |
| ivica zubac 2025 | 18.89 | 39.26 | 39.26 | 18.10 | **18.89** | 18.10 | 18.89 | −0.79 | **0.00** |
| aaron gordon 2026 | 33.66 | 19.60 | 19.60 | 31.98 | **33.66** | 19.60 | 19.60 | −14.06 | −14.06 |

Gordon is the one repaired row the clip cannot help: the champion *under*prices
him by $14.06M and a ceiling only ever pushes down. His repair matters for the
gate, not for the arm.

All **13 rows the clip moves**, repaired data, $M:

| row | pay | push | clip | \|err\| push → clip |
|---|---|---|---|---|
| marcus smart 2022 | 17.46 | 39.68 | 17.46 | 22.22 → 0.00 |
| ivica zubac 2025 | 18.89 | 39.26 | 18.89 | 20.37 → 0.00 |
| derrick white 2025 | 28.10 | 41.99 | 28.10 | 13.89 → 0.00 |
| lamarcus aldridge 2019 | 26.00 | 38.20 | 26.82 | 12.20 → 0.82 |
| jalen brunson 2025 | 34.94 | 46.39 | 34.94 | 11.45 → 0.00 |
| eric bledsoe 2019 | 15.62 | 25.44 | 18.00 | 9.82 → 2.37 |
| terry rozier 2022 | 21.49 | 26.69 | 21.49 | 5.20 → 0.00 |
| jrue holiday 2021 | 30.13 | 34.94 | 31.05 | 4.81 → 0.92 |
| toumani camara 2026 | 18.08 | 21.87 | 19.42 | 3.78 → 1.34 |
| wendell carter jr. 2026 | 18.10 | 20.90 | 18.10 | 2.79 → 0.00 |
| josh hart 2024 | 18.14 | 20.40 | 18.14 | 2.25 → 0.00 |
| dejounte murray 2024 | 29.52 | 30.32 | 29.52 | 0.80 → 0.00 |
| spencer dinwiddie 2019 | 10.61 | 10.71 | 10.53 | 0.11 → 0.07 |

$104.2M of absolute error removed across 13 rows; the top five carry $79.3M of it.

### How much is Smart alone

Additive to 1e-4 in both orders:

| order | step | ΔA1 |
|---|---|---|
| #23 (Smart) first | 0.7936 → 0.7987 | **+0.00511** |
| then #22 | 0.7987 → 0.7988 | +0.00013 |
| #22 first | 0.7936 → 0.7937 | +0.00013 |
| then #23 (Smart) | 0.7937 → 0.7988 | **+0.00511** |

**Smart alone is +0.00511 of A1**, against the architect's estimate of +0.0045 —
the claim holds, and it is 98% of the whole data-fix effect (+0.00524). In paired
terms he moves dSel from +0.00375 to +0.00824, i.e. he alone doubles the effect.
ISSUES #22's three rows are worth +0.00013 of A1 between them: they matter for the
hard gate, not for accuracy.

## 3. Gate verdicts — `push + told clip`, repaired data, tau = 0.52

| gate | bar | value | verdict |
|---|---|---|---|
| paired dSel t | t > 2 | **+1.11** (dSel +0.00856 ± 0.0077) | **FAIL** |
| counterweight brake | ≤ +$0.30M | −0.40 signed / **+0.40 in \|bias\|** | see below |
| 25%+ band brake | ≤ +$0.30M | +0.09 (1.06 → 1.15) | PASS |
| C2 worst fixed-row segment | ≤ +$0.30M | +0.02 (Early Bird) | PASS |
| B1 forward drop | ≤ 0.003 | **−0.0013** (0.8342 → 0.8355, forward R² *improves*) | PASS |
| hard gate: over-cap rows | ~0 of 156 | **0** (was 3) | PASS |

**The arm does not clear t > 2**, while A1 (+0.0123), A2 (+0.0176), MAE (−$0.160M)
and the max-zone MAE (−$1.118M) all improve and the forward number improves too.
Per the brief I state that and leave the adoption call to the architect; I am not
arguing for a gate change. The mechanism is not subtle: the arm moves 13 rows, 5
of them worth $79.3M of error, and 13 rows out of 944 cannot produce a stable
per-fold mean — fold 3 carries +0.0367 of a +0.0086 average.

**The counterweight brake needs a decision, because the two conventions in the
repo disagree on it.** `eval_max_branch_tau_sweep.py` computes this brake as a
*signed* growth (`bias_arm − bias_champion`), which reads −0.40 and passes; the C2
gate in the same script computes segment growth in *absolute* bias
(`|bias_arm| − |bias_champion|`), which for this band reads **+0.40 and fails**.
The band is underpredicted by the champion (−4.93) and the clip makes it more so
(−5.33). Two facts bear on which reading is right: the band's **MAE improves**,
$7.61M → $7.13M under the clip (this is the "judge a targeted intervention where
it acts" rule in CLAUDE.md), and only **2 of the 32 band rows are moved by the
clip at all** — so the −0.40 is two rows being pulled down to a legal ceiling, not
a surface shift. I have not picked a side; both numbers are in the table.

## 4. Anomalies

- **The +0.0123 pooled A1 gain is above the +0.01 red-flag line, and here is its
  one-sentence source**: 13 rows move, all 13 are selection rows, and five of them
  (Smart, Zubac, White, Brunson, Aldridge) each carried $11–22M of error that a
  hard CBA bound removes exactly. It is not a surface improvement and should never
  be quoted as one.
- **The confirmation split is untouched by this change** — zero of the 13 moved
  rows are confirmation rows, so the canary offers no independent read on the clip
  yet. Confirmation-only R² is identical for `push only` and `push + told clip`
  (0.8305, against the champion's 0.8070) — the clip changes nothing there. Worth
  knowing before the split is opened at a version bump.
- **Two of the five salary corrections are `inferred`, not `verified`, and the
  evidence is genuinely two-sided.** For Zubac 2025 and Gordon 2026 I recovered the
  prior base by inverting the CBA rule through observed pay (`pay / 1.40`), which
  is dollar-exact — but the *same two contracts* reconcile the other way against
  Spotrac's headline totals, which match the UNrepaired priors just as exactly
  (Zubac's $58.6M is the EAS-route schedule 18,102,000/19,550,160/20,998,320;
  Gordon's $103.6M is 1.40 × our old $22,841,455 with 8% raises, to $8,840 out of
  $103.6M). Under that reading our BBRef future-year salary rows are high instead
  and the priors were never stale. **This changes no number in this RESULT** — both
  readings give the identical ceiling for both rows — but it changes `prev_cap_pct`
  and possibly three future target rows. Filed as ISSUES #25 with the settling
  instrument (one Spotrac player page each). I did not pick a side silently: the
  brief adjudicated the renegotiation reading, so that is what is implemented, and
  `salary_corrections.csv` carries the counter-evidence in the row's own note.
- **One correction is a residual, by construction.** Smart 2022's `pay_above_base`
  of exactly $250,000 is observed pay minus the legal base, so that row's ceiling
  equals its pay. Sensitivity: dropping it entirely (ceiling $17.207M, a −$0.25M
  error on one row) changes A1 by less than 1e-5 and MAE by $0.0003M. Nothing
  rests on it.
- **Smart's prior repair rests on three agreeing instruments**, which is why it is
  `verified`: his 2018 4yr/$52M deal is an exact arithmetic sequence
  11,660,714 / 12,553,571 / 13,446,428 / **14,339,285** summing to $52,000,000 to
  the dollar and our table already carries its 2020-21 term; 1.20 × 14,339,285 =
  17,207,142 is $250k under his observed pay; and Spotrac carries 14,339,285
  (ISSUES #23). Separately, our 2019 figure for him ($11,768,879) fits neither the
  sequence nor anything else — I checked whether it was a league-wide 2019-20
  artifact by comparing every season-to-season salary ratio distribution in the
  table (2019→2020 median 1.0500 against 1.0603/1.0690/1.0741 for later pairs, same
  share inside the escalator band) and it is not. No row consumes it; noted in
  ISSUES #25.
- **`ext_cap_pct` still does not enter `max_eligible_pct` or the Stage-1 censor
  mask**, as required. Verified: `max_eligible_pct` is identical before and after
  to 5.6e-17, and 0 rows anywhere are paid above it.
- **ISSUES #21's entry title is now stale** ("raise caps are not implemented") —
  the infrastructure landed in 33325e9. Not my lane to rewrite; flagging it.

## 5. Files touched

| file | change |
|---|---|
| `src/model/extension_cap.py` | ISSUES #23 award anchor; `pay_above_base` allowance; `ext_addon_usd` column |
| `src/model/train.py` | `_load_salary_corrections()`; `prior_base` applied in `_load_prev_season_cap_pct` |
| `data/raw/raw_external/salary_corrections.csv` | **new** — 5 curated entries with derivation, source and confidence |
| `.gitignore` | whitelist for the above — `data/raw/raw_external/*` is ignored by default and the new table would have been silently dropped, which is the trap that file's own comment describes |
| `scripts/eval_told_clip.py` | **new** — the evidence harness for this RESULT |
| `ISSUES.md` | #22 and #23 deleted (fixed here); #25, #26, #27 added |
| `docs/briefs/2026-07-27-told-clip-and-data-fix.RESULT.md` | this file |

The harness is in `scripts/` rather than the scratchpad, matching
`eval_extension_cap.py` and `eval_route_mixture_p*.py`: every number above has to
reproduce from the landed tag. It reconstructs the pre-fix ceilings by
monkeypatching its *own* copies of the two loaders, so nothing in `extension_cap.py`
ships a "reproduce the bug" switch. Nothing in `route_mixture.py` or the floor
worker's lane was touched; `evaluate_suite.py` is unmodified.

## 6. Proposed commit message

```
Fix ISSUES #22/#23 and re-measure the told-route Stage-3 clip

The designated-veteran test granted Marcus Smart a 2022 supermax exemption from
a DPOY he won nine months AFTER signing, because the award window was unioned
over the signing and paying seasons. The award anchor is now the signing season
alone; the service-year anchor stays unioned, since it is our debut-based count
that is unreliable, not the calendar. One row flips, and his extension ceiling
falls from the $43.28M tier to his real $17.46M raise cap.

The three over-cap rows are repaired through a curated salary_corrections.csv
carrying two kinds of entry: prior_base, the contractual base salary our table
does not hold (applied once, inside _load_prev_season_cap_pct), and
pay_above_base, money inside observed pay that the raise cap does not govern —
Murray's trade bonus, pinned independently by his $114.24M contract total. No
ceiling is clamped to observed pay by rule, and each entry carries its evidence
and a confidence field; the two renegotiate-and-extend priors are marked
inferred, with the contradicting Spotrac totals filed as ISSUES #25.

Over-cap rows go 3 -> 0 with the frame invariant: 944 rows, targets and
max_eligible_pct identical, route labels identical, so the champion and both
classifiers are unchanged and only four ceilings move.

Re-measured at the pre-registered tau = 0.52, the told clip scores A1 0.7988 /
MAE $2.916M / max-zone MAE $3.179M against the champion's 0.7865 / $3.076M /
$4.297M, with the 25%+ band brake at +0.09 and forward R2 improving 0.8342 ->
0.8355. Paired dSel is +0.00856 at t = 1.11 -- short of the t > 2 bar, on 13
moved rows. Smart alone is +0.00511 of the A1 gain.

These numbers use the told-route convention adopted 2026-07-26: the route is
scored as an available covariate, in the same column as the ex-ante numbers.
v7.1x-v7.13x predate that convention and are not naively comparable.
```

## 7. ISSUES.md changes

Deleted **#22** and **#23** — both fixed on this branch, with the caveats promoted
to their own entries rather than left inside a closed issue. Added:

- **#25** — Spotrac's contract totals disagree with our salary schedules on three
  extensions; the reason two corrections are `inferred`. Includes the settling
  instrument and what to flip if it goes the other way.
- **#26** — the stored `prev_cap_pct` feature still carries pre-correction values
  for three rows. Corrections are applied at the loader; the feature is baked into
  `training_data_v2.csv` by `phase3.py`. Deliberately not rebuilt here, so that the
  re-measure could isolate the clip with a bit-identical champion.
- **#27** — Smart 2022 still reads a 35% supermax ceiling in `max_eligible_pct`.
  `_compute_max_eligible` has no signing-date instrument, so the #23 fix lands only
  in `extension_cap`. Fixing it moves the champion and belongs to a version bump.
