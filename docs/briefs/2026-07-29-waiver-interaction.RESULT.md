# RESULT — a waiver erases the price history: the Stage-1 interaction

Date: 2026-07-29
Pin: `master` @ d62179e (v8.2x, A1 0.8210 / A2 0.8625 / B1 0.8528, 944 rows)
Branch: `worker/waiver-interaction`

## Verdict

**Arm A (`prev_cap_pct_x_waived`): do not adopt.** Paired ΔSel **+0.00334 ± 0.00233,
t = 1.43**, below the t > 2 bar.

**Arm B (+ `mpg_x_waived`): do not adopt.** Paired ΔSel **+0.00539 ± 0.00275,
t = 1.96**, below the t > 2 bar.

Both arms move every number in the right direction and damage nothing — A2, B1,
calibration, and eight of nine C2 segments all improve, and a permuted-waiver
placebo confirms the effect is real content rather than tree perturbation (the
placebo scores 0). They fail on statistical **power**, not on being spurious:
the effect is the right size and too fold-unstable to clear t > 2. Part 0 shows
this was decided before either arm was built: **an oracle that predicts all
eleven top-band rows perfectly scores ΔSel +0.00546 at t = 1.09.** The ceiling
itself does not clear the gate. What limits this segment is not the model's grip
on it but the arithmetic of eleven rows, four of them locked in the confirmation
split, spread across five folds.

The measurement that matters most from this session is not either arm. It is
that **three training rows carry a waiving team's stretched dead money as if it
were the player's salary** (ISSUES #32), and one of them is the largest waived
salary in the frame.

---

## Part 0 — the headroom, measured before anything was built

All figures are the v8.2x champion's own OOF column on the 944-row frame,
reproduced from the pinned tag. The reproduction is exact: A1 0.8210, calibration
slope 0.9709, and the incumbent arm's `fold_r2_selection` matrix matches
`outputs/models/evaluation_suite.json` **bit-for-bit** (max abs diff 0.00e+00),
so every paired delta below is as tightly paired as the protocol allows.

The brief's three tables reproduce exactly. Slopes of this year's cap_pct on
`prev_cap_pct`: not waived (n=822) **+0.750** (Spearman +0.594); waived (n=122)
**+0.140** (Spearman +0.429). Prior pay is near-identical in aggregate (0.0619
waived vs 0.0656 not). Landing spots and the champion's per-band bias reproduce
to the cent under the brief's convention, `mean(cap_pct) x the 2026 cap`.

### 0.1 What share of squared error do the eleven rows carry?

Two definitions, because they differ and the brief's Lillard figure is the
second one:

| definition | Lillard 2025 | all 11 rows |
|---|---:|---:|
| cap_pct², all 944 rows | 2.18% | 6.88% |
| **dollars², all 944 rows** | **3.20%** | **7.27%** |
| cap_pct², selection rows only | 2.50% | 2.75% |
| dollars², selection rows only | 3.60% | 3.83% |

Individually, on the 944-row frame (cap_pct² share of total SSE):

| player | season | prior $M | actual $M | predicted $M | error $M | SSE share | fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| kemba walker | 2021 | 35.41 | 8.73 | 27.81 | +19.08 | 2.54% | 2 \* |
| damian lillard | 2025 | 53.67 | 14.10 | 38.42 | +24.32 | 2.18% | 3 |
| andre drummond | 2021 | 28.80 | 2.40 | 14.49 | +12.08 | 1.02% | 4 \* |
| russell westbrook | 2023 | 50.92 | 3.84 | 15.80 | +11.96 | 0.68% | 2 \* |
| bradley beal | 2025 | 55.22 | 19.38 | 11.51 | −7.88 | 0.23% | 3 \* |
| nicolas batum | 2020 | 25.57 | 8.86 | 4.44 | −4.42 | 0.14% | 1 |
| deandre ayton | 2025 | 37.41 | 8.10 | 12.31 | +4.20 | 0.07% | 1 |
| dwight howard | 2019 | 25.99 | 2.56 | 3.29 | +0.72 | 0.00% | 3 |
| chris paul | 2024 | 31.83 | 10.46 | 10.06 | −0.40 | 0.00% | 3 |
| kevin love | 2023 | 30.19 | 3.84 | 4.03 | +0.20 | 0.00% | 1 |
| kyle lowry | 2024 | 30.68 | 2.09 | 2.09 | −0.00 | 0.00% | 3 |
| **group** | | | | | | **6.88%** | |

\* = locked confirmation row.

**Four of the eleven are confirmation rows, and they include three of the four
largest errors** (Walker, Drummond, Westbrook — 4.24 of the 6.88 points). The
decision metric cannot see them. That single fact is why the pooled headroom and
the decidable headroom differ by more than a factor of two.

### 0.2 The oracle bound

Substituting the band's observed mean for the champion's prediction on those
rows, and reading the result the way the gate reads it — paired per-fold R² on
the selection pool:

| oracle | pooled A1 | ΔSel | t | per fold |
|---|---:|---:|---:|---|
| champion | 0.8210 | — | — | — |
| band mean on all 11 | 0.8313 (+0.0103) | **+0.00435** | **+1.05** | [0, +0.0008, 0, +0.0210, 0] |
| band mean on the 10 excl. Lillard | 0.8276 (+0.0066) | −0.00040 | −0.66 | [0, +0.0008, 0, −0.0028, 0] |
| perfect prediction on all 11 | — | **+0.00546** | **+1.09** | [0, +0.0019, 0, +0.0254, 0] |
| perfect prediction on the 10 excl. Lillard | — | +0.00039 | +1.03 | [0, +0.0019, 0, +0.0001, 0] |
| perfect prediction on Lillard alone | — | +0.00507 | +1.00 | [0, 0, 0, +0.0254, 0] |
| perfect prediction on all 122 waived rows | — | +0.01138 | +2.01 | [+0.0076, +0.0045, +0.0022, +0.0336, +0.0091] |

**Say it plainly, as the brief asked.** The pooled A1 headroom is +0.0103, which
looks like room. The gate-comparable headroom is **+0.00546 at t = 1.09**, and it
fails gate 1 — not on magnitude but on concentration. Excluding Lillard the
ceiling is +0.00039, i.e. nothing: **Lillard 2025 alone is 93% of the entire
oracle**. A term that cannot beat this bound cannot clear the gate, and no term
can beat it.

Even an oracle over the whole 122-row waived population reaches only t = 2.01 —
at the bar, not past it, and that oracle is unreachable by construction.

**This measurement stands on its own and did not need either arm.** The eleven
rows are a real and well-identified pricing error worth about $5.4M of bias, and
they are simultaneously too few, too unevenly folded, and too heavily
confirmation-locked to be adoptable through the selection gate. Those are
different statements and both are true.

---

## Part 0b — Bradley Beal 2025 is a fabricated observation, and he is not alone

**Verdict: the $19.383M row is Phoenix's stretched dead-money obligation, not a
contract Beal signed.** Five independent instruments agree, and none dissent:

1. `salaries.csv` carries Beal at team **PHO** with **$19,383,010 in each of
   2025, 2026, 2027, 2028 and 2029** — five byte-identical years, from
   `bbref_team_contracts`. 5 × $19,383,010 = **$96,915,050**, the reported ~$97M
   stretch. A negotiated contract carries CBA raises; five identical years to the
   cent is a stretch annuity, not a price.
2. He was waived by the team the row bills: *"Waived by Phoenix (PHX) via Buyout
   and Stretch Provision"*, 2025-07-16.
3. `contract_signing_dates.csv` records the contract he actually signed:
   **2025-07-18, LAC, 2 years / $10,980,000**.
4. `spotrac_signing_types.csv` prices that deal at a non-taxpayer MLE,
   2yr/$10,975,700, **AAV $5,487,850**.
5. `contract_structure_v2.csv` read the flat five-year stretch schedule as a
   newly signed five-year contract and stamped 2025 `year_in_contract=1`, which
   is the only reason the row reaches the evaluation frame at all.

So the row sits in training at 12.53% of cap when the price the market set was
about 3.3%. Per the brief I have **not fixed it**; it is filed as ISSUES #32 with
a reproducer, and every arm below is reported both with the row and without it.

### The scan found two more

The brief noted Lillard 2025 is correct ($14.104M = the Portland deal) and asked
whether the defect is systematic. It is more systematic than assumed. A new
reproducer, `scripts/audit_stretched_salaries.py`, compares every evaluation
row's salary against the AAV of the Spotrac contract block that best matches it,
and requires a waiver *preceding the row* to corroborate a disagreement:

| row | our salary | what it is | what he signed |
|---|---:|---|---|
| bradley beal 2025 | $19,383,010 | PHX stretch, 5 × $19,383,010 = $96.9M | LAC 2yr/$10.98M (AAV $5.49M) |
| joakim noah 2020 | $6,431,667 | NYK stretch, 3 × $6,431,667 = $19,295,001 exactly | LAC minimum (AAV $1.49M) |
| nicolas batum 2020 | $8,856,969 | CHA dead money, repeated in 2020 and 2021 | LAC 1yr/$2.56M |

Noah's arithmetic is exact against the transaction text (*"Waived by New York
(NYK) - Stretch Provision on 2019-2021"*) and his salary row is billed to **NYK**
for a season he did not play there. Batum's $8,856,969 is billed to Charlotte,
who waived him, and repeats identically in 2020 and 2021; it is $26.57M over
three years against the ~$27.13M he was owed, so I could not recover the exact
base from our sources — the attribution and the repetition establish the
mechanism, the cent-level arithmetic does not. I have flagged that limit rather
than asserted a number.

Two further rows flag on the salary/AAV ratio and are **not** this defect —
Markkanen 2024 (a legal renegotiate-and-extend, so the block predates the row's
own contract) and Winslow 2019 (an extension year matched against the original
rookie block; his waivers are in 2023 and 2024, after the row). The script
separates them explicitly, and an earlier looser version that keyed on "waived at
any point" wrongly flagged Winslow — worth knowing before the rule is reused.

Batum 2020 is one of the eleven top-band rows. So **two of the eleven rows in the
band this feature targets carry a fabricated target**, which contaminates the
band mean the Part 0 oracle aims at.

---

## The NaN decision, registered before scoring

**Native NaN propagation: a NaN times a number is NaN.** Where Spotrac does not
cover the row we do not know whether the player was waived, so we do not know the
product either, and writing 0 would assert "not waived" — a different claim.
`attach_waiver_interactions` therefore computes `prev_cap_pct * is_waived`
directly and lets NaN propagate. The evaluation frame's existing median fill then
applies to this column exactly as it applies to every other feature; the fill is
inherited, not bypassed, which is what makes this the live pipeline's behaviour.

Verified on the emitted table: 3,113 rows, **819 NaN in the product, the same 819
rows that are NaN in `is_waived`**, 227 non-zero, 2,067 zero, and the column
equals `prev_cap_pct * is_waived` exactly. The median fill maps the 819 unknowns
to 0, because the median of the known values is 0 — so on this data native-NaN
and a median-filled ship form coincide numerically. **That is a measured
coincidence, not a reason the distinction does not matter**: it holds only while
fewer than half the known rows are waived, and the brief's own precedent
(+0.0046 native vs +0.0001 median-filled) is why it was checked rather than
assumed.

## Ship form

The scored column is byte-for-byte what `build_waiver_features` emits.
`scripts/build_waiver_features.py --interactions` writes
`prev_cap_pct_x_waived` and `mpg_x_waived` into `training_data_v2.csv`; the
harness then loads them through the ordinary `load_evaluation_frame` path with no
special handling. The emitter re-reads the table, widens it, and **hard-fails if
any pre-existing column moves** — verified: rows 3,113 → 3,113, two columns
added, zero columns removed, zero pre-existing columns changed.

The products must be written after stage 3, not inside it: they need
`prev_cap_pct`, which `scripts/phase3.py` adds, so there is no earlier point at
which both factors exist. Landing this would append one line to
`scripts/rebuild_training_data.py` and two to `src/model/predict.py` (after the
`prev_cap_pct` borrow at line 107); neither is done here, since the verdict is
do-not-adopt.

---

## Numbers

Every arm is the v8.2x champion's 15 features plus the listed additions, scored
through `run_suite` with the champion composition, 5 player-grouped folds x 10
seeds. Reference: `scripts/eval_waiver_interaction.py`.

### Arms, 944 rows (Beal in)

| arm | added | A1 | A2 | B1 | slope | MAE $M |
|---|---|---:|---:|---:|---:|---:|
| incumbent | — | 0.8210 | 0.8625 | 0.8528 | 0.9709 | 2.74 |
| coverage | `is_waived_known` | 0.8212 | 0.8613 | 0.8587 | 0.9698 | 2.74 |
| **Arm A** | `prev_cap_pct_x_waived` | 0.8244 | 0.8648 | 0.8606 | 0.9702 | 2.72 |
| Arm A over coverage | + `is_waived_known` | 0.8253 | 0.8650 | 0.8613 | 0.9722 | 2.71 |
| **Arm B** | + `mpg_x_waived` | 0.8266 | 0.8653 | 0.8619 | 0.9712 | 2.70 |
| Arm B over coverage | + `is_waived_known` | 0.8262 | 0.8642 | 0.8622 | 0.9710 | 2.70 |

### Paired deltas on the selection pool, 944 rows

| comparison | ΔSel | ± SE | t | ΔA2 | ΔB1 | C1 excess | worst C2 |
|---|---:|---:|---:|---:|---:|---:|---|
| **Arm A** vs incumbent | **+0.00334** | 0.00233 | **+1.43** | +0.0023 | +0.0078 | +0.0006 | Non-Bird +0.052M |
| Arm A over coverage | +0.00404 | 0.00195 | +2.07 | +0.0037 | +0.0026 | −0.0023 | Other +0.023M |
| **Arm B** vs incumbent | **+0.00539** | 0.00275 | **+1.96** | +0.0028 | +0.0090 | −0.0004 | Non-Bird +0.010M |
| Arm B over coverage | +0.00448 | 0.00252 | +1.77 | +0.0029 | +0.0035 | −0.0012 | Minimum −0.018M |
| coverage alone | +0.00025 | 0.00039 | +0.64 | −0.0012 | +0.0058 | +0.0010 | Non-Bird +0.043M |

Per-fold ΔSel, in full (the brief asked for these explicitly):

| comparison | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---:|---:|---:|---:|---:|
| Arm A | −0.00379 | +0.00797 | −0.00060 | +0.00702 | +0.00608 |
| Arm A over coverage | −0.00066 | +0.00680 | −0.00071 | +0.00833 | +0.00644 |
| Arm B | −0.00117 | +0.01042 | −0.00128 | +0.01107 | +0.00788 |
| Arm B over coverage | −0.00138 | +0.00873 | −0.00123 | +0.01094 | +0.00533 |
| coverage alone | −0.00090 | +0.00146 | +0.00025 | −0.00012 | +0.00056 |

Where the eleven top-band rows fall, and how many the decision metric can see:

| fold | top-band rows | of those, confirmation | **visible to ΔSel** |
|---:|---:|---:|---:|
| 0 | 0 | 0 | **0** |
| 1 | 3 (Ayton, Love, Batum) | 0 | **3** |
| 2 | 2 (Walker, Westbrook) | 2 | **0** |
| 3 | 5 (Beal, Paul, Lillard, Howard, Lowry) | 1 (Beal) | **4** |
| 4 | 1 (Drummond) | 1 | **0** |

No single fold carries the gain — folds 1, 3 and 4 all contribute — so the
brief's stated red flag does not fire. **A different one does**: fold 4 posts
+0.00608 while containing zero top-band rows the metric can see. See Anomalies.

### The same battery without Beal, 943 rows

R² is not comparable across the two row sets (D1), so the incumbent's A1 is
restated per row set and only the paired deltas are compared.

| arm | A1 | A2 | B1 | slope |
|---|---:|---:|---:|---:|
| incumbent | 0.8251 | 0.8656 | 0.8528 | 0.9723 |
| Arm A | 0.8288 | 0.8667 | 0.8612 | 0.9721 |
| Arm B | 0.8314 | 0.8680 | 0.8629 | 0.9743 |

| comparison | ΔSel | ± SE | t | ΔA2 | ΔB1 | C1 excess | per fold |
|---|---:|---:|---:|---:|---:|---:|---|
| **Arm A** | **+0.00220** | 0.00162 | **+1.36** | +0.0011 | +0.0084 | +0.0002 | [−0.0017, +0.0062, −0.0014, +0.0030, +0.0050] |
| Arm A over coverage | +0.00241 | 0.00178 | +1.36 | +0.0019 | +0.0031 | −0.0021 | [−0.0015, +0.0052, −0.0023, +0.0048, +0.0058] |
| **Arm B** | **+0.00371** | 0.00241 | **+1.54** | +0.0024 | +0.0100 | −0.0020 | [−0.0003, +0.0084, −0.0036, +0.0059, +0.0082] |
| Arm B over coverage | +0.00350 | 0.00218 | +1.60 | +0.0019 | +0.0041 | −0.0010 | [−0.0021, +0.0077, −0.0015, +0.0060, +0.0074] |
| coverage alone | +0.00018 | 0.00037 | +0.48 | −0.0002 | +0.0061 | −0.0001 | [+0.0009, +0.0008, −0.0005, −0.0009, +0.0006] |

**Removing the fabricated row lowers both arms** (A +0.00334 → +0.00220, B
+0.00539 → +0.00371). That is not a defence of the row: Beal is a confirmation
row, so it cannot enter ΔSel directly and can only act through the training
folds. The verdict is unchanged on either row set, which is the point of running
both.

### Fixed-row segments, 944 rows — every arm on the INCUMBENT's row groups

Cells are bias / MAE in $M.

| segment | n | incumbent | Arm A | Arm B |
|---|---:|---:|---:|---:|
| waived | 122 | +0.69 / 1.80 | +0.56 / 1.74 | +0.44 / 1.64 |
| **not waived / unknown** | **822** | **−0.24 / 2.88** | **−0.21 / 2.86** | **−0.20 / 2.86** |
| waived prior <3% | 66 | +0.21 / 0.96 | +0.15 / 0.94 | +0.06 / 0.86 |
| waived prior 3-7% | 21 | +0.70 / 1.00 | +0.57 / 0.93 | +0.55 / 0.91 |
| waived prior 7-12% | 15 | −0.54 / 2.03 | −0.59 / 2.06 | −0.78 / 1.99 |
| waived prior 12-20% | 9 | +0.46 / 2.14 | +0.19 / 2.02 | +0.12 / 1.76 |
| **waived prior ≥20%** | **11** | **+5.44 / 7.75** | **+4.93 / 7.42** | **+4.44 / 7.10** |

The brief's test — *a term that only fires on 122 rows must not move the other
822* — **fails, and that is the finding.** The 822 move: bias −0.24 → −0.21,
MAE 2.88 → 2.86, and 624 of them move by more than $0.01M. See Anomalies.

Without Beal (943 rows) the targeted effect is larger, because he was the one
top-band row the champion *under*-priced and he was masking the rest:

| segment | n | incumbent | Arm A | Arm B |
|---|---:|---:|---:|---:|
| waived | 121 | +0.64 / 1.71 | +0.50 / 1.65 | +0.38 / 1.51 |
| waived prior ≥20% | 10 | +6.07 / 7.59 | +5.28 / 6.92 | +4.82 / 6.29 |

### The eleven top-band rows, predicted $M, 944 rows

| player | season | actual | incumbent | Arm A | Arm B | fold |
|---|---:|---:|---:|---:|---:|---:|
| andre drummond | 2021 | 2.40 | 14.49 | 13.85 | 13.09 | 4 \* |
| bradley beal | 2025 | 19.38 | 11.51 | 10.98 | 9.97 | 3 \* |
| chris paul | 2024 | 10.46 | 10.06 | 9.71 | 9.56 | 3 |
| **damian lillard** | **2025** | **14.10** | **38.42** | **36.95** | **36.01** | 3 |
| deandre ayton | 2025 | 8.10 | 12.31 | 10.50 | 9.59 | 1 |
| dwight howard | 2019 | 2.56 | 3.29 | 3.88 | 3.74 | 3 |
| kemba walker | 2021 | 8.73 | 27.81 | 26.93 | 26.13 | 2 \* |
| kevin love | 2023 | 3.84 | 4.03 | 4.17 | 4.57 | 1 |
| kyle lowry | 2024 | 2.09 | 2.09 | 2.21 | 2.09 | 3 |
| nicolas batum | 2020 | 8.86 | 4.44 | 4.34 | 4.55 | 1 |
| russell westbrook | 2023 | 3.84 | 15.80 | 15.11 | 13.91 | 2 \* |

\* = locked confirmation row.

**Lillard 2025 specifically**: $38.42M → $36.95M (Arm A) → $36.01M (Arm B)
against an actual of $14.10M. The registered form moves him **$1.47M**, and Arm B
$2.41M, on a $24.32M error. The mechanism is real and directionally right; its
magnitude on the case that motivated the brief is about 6% of the gap. A tree
learning an average discount off 122 rows cannot do more, which is the same
constraint `is_waived` already hit — the interaction relaxes it, it does not
remove it.

### C2 mechanism table, 944 rows (bias $M, and growth in |bias|)

| mechanism | n | incumbent | Arm A | Δ\|bias\| A | Arm B | Δ\|bias\| B |
|---|---:|---:|---:|---:|---:|---:|
| Bird Rights | 274 | −1.767 | −1.630 | −0.137 | −1.610 | −0.157 |
| Minimum | 242 | +1.693 | +1.707 | +0.014 | +1.678 | −0.015 |
| Unknown | 131 | +0.259 | −0.075 | −0.185 | −0.101 | −0.158 |
| MLE | 123 | +0.517 | +0.498 | −0.018 | +0.467 | −0.050 |
| Cap Space | 76 | −0.590 | −0.504 | −0.085 | −0.523 | −0.066 |
| Early Bird | 46 | −1.527 | −1.346 | −0.181 | −1.308 | −0.219 |
| Non-Bird | 20 | +1.718 | +1.770 | **+0.052** | +1.728 | +0.010 |
| Other | 18 | −0.097 | −0.086 | −0.011 | −0.069 | −0.028 |
| Sign & Trade | 13 | −4.450 | −4.232 | −0.218 | −4.127 | −0.323 |

Worst growth is +$0.052M (Arm A) and +$0.010M (Arm B), far inside the $0.30M
guard. Eight of nine segments improve. Nothing here argues against either arm.

---

## Gate verdicts

| # | gate | Arm A | Arm B |
|---|---|---|---|
| 1 | ΔSel ≥ +0.002 **and t > 2** | +0.00334, **t 1.43 — FAIL** | +0.00539, **t 1.96 — FAIL** |
| 2 | A2 moves the same direction | +0.0023 — pass | +0.0028 — pass |
| 3 | B1 forward drop ≤ 0.003 | +0.0078 (improves) — pass | +0.0090 (improves) — pass |
| 4 | C2 fixed-segment \|bias\| growth ≤ $0.30M | +$0.052M — pass | +$0.010M — pass |
| 5 | C1 relative excess ≤ 0.005 vs 0.9709 | +0.0006 — pass | −0.0004 — pass |
| 6 | coverage-indicator increment | +0.00404, t 2.07 — pass | +0.00448, t 1.77 — pass |

**Both arms fail gate 1 and pass every other gate.** Both fail on the *t*, not on
the magnitude: the effect is the right size and the wrong shape, spread over
folds whose disagreement is larger than the effect.

On gate 6: the coverage channel does not explain either arm. Coverage alone is
+0.00025 (t 0.64), reproducing the v8.2x RESULT's +0.00029 (t 1.02) on the same
frame, and each arm's increment *over* a coverage arm is as large as its raw
delta. Note the increment reading for Arm A reaches t = 2.07 — but that arm also
carries `is_waived_known`, which v8.2x deliberately declined to ship because it
encodes source availability rather than a basketball fact. It is a control, not a
shippable form, and the registered Arm A is the 16-feature model at t = 1.43.

---

## Anomalies

**1. Most of the measured gain is not the mechanism.** This is the important one.
Decomposing the selection-pool change in squared error by row group
(`--save-oof`, then the group split):

| group | n | rows moved > $0.01M | mean \|move\| $M | share of ΔSSE, Arm A |
|---|---:|---:|---:|---:|
| waived, prior ≥20% | 11 | 11 | 0.664 | **22.7%** |
| waived, prior <20% | 111 | 51 | 0.106 | 15.7% |
| **not waived / unknown** | **822** | **624** | **0.212** | **61.6%** |
| — of which source-known | 670 | 517 | 0.165 | 7.8% |
| — **of which source-unknown** | **152** | **107** | **0.419** | **53.8%** |

The eleven rows the term was designed for supply **22.7%** of Arm A's improvement.
**53.8% comes from the 152 rows whose `is_waived` is unknown** — rows where the
new column is a median-filled constant 0 and carries no information whatsoever,
and which nonetheless move by a mean of $0.419M, more than twice the movement of
the source-known non-waived rows. Arm B shows the same shape (21.3% top band,
54.4% non-waived).

Per fold, split into the top band and everything else, this is unambiguous:

| fold | top-band rows visible | ΔSSE total | from the top band | from the rest |
|---:|---:|---:|---:|---:|
| 0 | 0 | +0.005659 | 0 | +0.005659 |
| 1 | 3 | −0.010610 | −0.000421 | −0.010189 |
| 2 | 0 | +0.000791 | 0 | +0.000791 |
| 3 | 4 | −0.006816 | −0.002789 | −0.004027 |
| 4 | 0 | −0.003157 | 0 | −0.003157 |

Fold 4's entire −0.003157 comes from rows outside the band, and folds 1 and 3 —
which hold every visible top-band row — still take most of their gain from
elsewhere. The largest single movers are not waived players at all: Cam Thomas
2025 ($20.66M → $17.95M, error 14.67 → 11.96), Herbert Jones 2023 (3.81 → 1.11),
and against them Nic Claxton 2024 (10.92 → **13.46**) and Bruce Brown 2023
(8.22 → **10.65**) get worse. All have `prev_cap_pct_x_waived` exactly 0.

The tempting reading is tree perturbation — a 16th column reshuffles splits and
column subsampling throughout the ensemble, moving predictions on rows the column
says nothing about — and that this diffuse gain is therefore luck. **Three placebo
arms refute that reading.** Each is `prev_cap_pct` times a *permuted* `is_waived`:
identical column shape, identical value distribution, identical marginal waiver
rate, and zero content, because the permutation severs the link to who was
actually waived.

| arm | ΔSel | ± SE | t |
|---|---:|---:|---:|
| Arm A (real) | +0.00334 | 0.00233 | +1.43 |
| placebo, seed 0 | −0.00035 | 0.00061 | −0.58 |
| placebo, seed 1 | +0.00005 | 0.00095 | +0.05 |
| placebo, seed 2 | −0.00016 | 0.00037 | −0.42 |

A contentless column of this exact shape scores **zero** (all |t| < 0.6), and on
the fixed-row segments the placebos leave the ≥20% band untouched (+5.44 →
+5.42/+5.56/+5.48 vs Arm A's +4.93). So the diffuse gain is **not** perturbation
and **not** luck: it is real content — the tree carving the genuinely-waived
high-`prev` rows into their own low-prediction region — whose benefit then spills
onto other rows through the shared model and base rate. That reflects well on the
mechanism. It does not change the verdict, because a real effect this small and
this fold-unstable still fails the t > 2 gate; but it does correct the framing:
**Arm A fails on statistical power, not on being spurious.**

The revised summary of Anomaly 1, then: the mechanism is genuine (the placebo
proves it), the top-band rows it was designed for carry only 22.7% of its effect
(the decomposition proves it), and the whole thing is too small relative to
fold noise to adopt (gate 1). All three are true at once.

**2. Two of the eleven target rows carry a fabricated target.** Beal 2025 and
Batum 2020 are both stretch-provision artifacts (ISSUES #32). The band mean the
Part 0 oracle aims at is therefore contaminated, and Beal in particular pulls the
band's observed mean *up* by about $1.2M, which flatters the champion's bias on
the band and understates the real over-pricing. The 943-row segment table shows
this directly: incumbent bias on the band is +5.44 with Beal and **+6.07**
without him.

**3. The attenuation caveat in the brief does not survive testing — the effect is
real.** The brief attached a caveat to the feature-attenuation table (mpg
Spearman 0.776 → 0.320): that the waived group's compressed target with heavy
ties makes rank correlation attenuate *mechanically*. It does not, for a reason
worth recording: **Spearman is invariant to any strictly monotone transform of
the target, so compression alone cannot move it** — only ties can. Two controls:

- *Tie control.* Quantile-map the 822 not-waived rows' target onto the waived
  group's empirical distribution, reproducing its spread (sd 0.0193 vs 0.0196)
  and its exact 72 distinct values while preserving rank order perfectly. mpg's
  Spearman goes 0.776 → **0.775**. The mechanical component is **0.001** against
  an observed attenuation of 0.456.
- *Sample-size control.* 4,000 draws of 122 not-waived rows: mpg's null is
  +0.772 ± 0.042, 95% interval [+0.685, +0.846]. The observed +0.320 sits about
  ten standard deviations below it, p < 0.0001. `prev_cap_pct` p = 0.0055;
  `darko_dpm_z` and `lebron_z` p < 0.0001.

So Arm B's premise is *better* founded than the brief allowed — every feature
genuinely predicts waived players' pay less well. This does not rescue Arm B,
which still fails gate 1, but the caveat should not be carried forward as stated.

**4. No delta exceeded +0.01, so no re-verify flag fired.** The largest paired
delta anywhere in the battery is Arm B's +0.00539.

**5. A latent data-loss bug, found and fixed.** `scripts/build_waiver_features.py`
parsed an absent HTML cache to an empty table and wrote it over
`data/processed/spotrac_transactions.csv`, destroying 6,849 dated transactions
and zeroing `is_waived` — the "a failed fetch degrades to stale, never to
missing" rule, violated. It triggers in any fresh worktree, because the cache is
gitignored and the parsed table is tracked. It bit this session; the table was
restored from git and the script now refuses to shrink the artifact, with
`--interactions` runnable standalone. No ISSUES entry, because it is fixed on
this branch.

---

## The Stage-2 temptation, declined

The brief asked that any pull back toward a clip be written down rather than
implemented. There was one, and it should be recorded because the numbers look
seductive: on the 943-row frame the ≥20% band's incumbent bias is **+$6.07M**
across ten rows, and no waived row in the frame exceeds 15% of the cap. A clip at
that line reads well in sample for exactly the reason it is forbidden — it is
fitted to a 122-row sample maximum.

I did not implement it, and the Part 0 numbers are an independent argument
against it that did not exist when the brief was written: **even a perfect
oracle on these rows scores t = 1.09.** A clip cannot beat an oracle. So the
Stage-2 form would buy less than +0.0055 of ΔSel while permanently forbidding the
model from ever pricing a bought-out star correctly — and Beal 2025, one of the
rows it would clip, turns out not to be a real observation at all. The standing
decision (Stage 2 holds deterministic CBA bounds on the player himself; no CBA
provision caps a waived player's next contract) is correct and this session
found nothing to reopen it.

---

## Files touched

| file | change |
|---|---|
| `src/features/waiver_history.py` | `WAIVER_INTERACTIONS`, `attach_waiver_interactions()` — the products, native NaN |
| `scripts/build_waiver_features.py` | `emit_interactions()` + `--interactions`; **fix**: refuse to shrink `spotrac_transactions.csv` |
| `scripts/eval_waiver_interaction.py` | new — the six-arm paired battery, fixed-row segments, placebo arms |
| `scripts/audit_stretched_salaries.py` | new — the ISSUES #32 reproducer |
| `tests/test_waiver_history.py` | 5 tests for the interaction and its NaN semantics (8 pass) |
| `ISSUES.md` | entry **32**; header's "highest ever used" 31 → 32 |
| `data/processed/training_data_v2.csv` | 2 columns added, 3,113 rows, no pre-existing column moved |
| `docs/briefs/2026-07-29-waiver-interaction.RESULT.md` | this file |

**`src/model/train.py` is untouched.** `FEATURE_COLS` — the one line the brief
flagged as shared with the other workers — needs no merge, because the verdict is
do-not-adopt. The harness builds each arm's feature list itself.

Outputs: `outputs/models/waiver_interaction_eval.json`,
`waiver_interaction_eval_no_beal.json`, `waiver_interaction_oof.json` +
`_oof.csv`, `waiver_interaction_placebo.json`, and the run logs beside them.

## Reproducing

```
python scripts/build_waiver_features.py --interactions
OMP_NUM_THREADS=6 python scripts/eval_waiver_interaction.py
OMP_NUM_THREADS=6 python scripts/eval_waiver_interaction.py --drop-beal
OMP_NUM_THREADS=6 python scripts/eval_waiver_interaction.py --placebo
python scripts/audit_stretched_salaries.py
```

About 40 minutes per six-arm battery on six threads.

---

## Proposed commit message

```
Measure the waiver interaction; adopt neither arm, and find three fabricated salaries

prev_cap_pct_x_waived scores +0.00334 (t=1.43) and the mpg extension
+0.00539 (t=1.96) on the selection pool. Both fail the t>2 bar and pass
every other gate: A2 and B1 move with the pooled gain, worst C2 growth is
$0.052M against a $0.30M guard, calibration excess +0.0006, and the
coverage control is +0.00025 (t=0.64), so the Spotrac channel explains
none of it. The verdict holds on 943 rows with the Beal row removed.

The measurement that decides it was taken before either arm was built.
An oracle that predicts all eleven top-band waived rows PERFECTLY scores
+0.00546 at t=1.09 — the ceiling on this segment does not clear the gate.
Four of the eleven are confirmation rows, including three of the four
largest errors, and Lillard 2025 alone is 93% of what is left.

A decomposition says the arms are not even doing what they were built to
do: 22.7% of Arm A's improvement lands on the eleven target rows and 53.8%
on the 152 rows whose waiver status is unknown, where the column is a
constant with no information in it.

Also: three training rows carry a waiving team's stretched dead money as
if it were the player's salary — Beal 2025 ($19.383M is Phoenix's ~$97M
stretch, not the 2yr/$10.98M he signed with the Clippers), Noah 2020 and
Batum 2020. Filed as ISSUES #32 with scripts/audit_stretched_salaries.py
as the reproducer; not fixed here.

Fixes a latent data-loss bug in build_waiver_features.py, which wrote an
empty transaction table over a good one whenever the gitignored HTML cache
was absent.
```

## ISSUES.md additions

**#32** — three training rows carry a waiving team's stretched dead money as if
it were the player's salary (Beal 2025, Noah 2020, Batum 2020), with
`scripts/audit_stretched_salaries.py` as the reproducer and the exact signed
figures for all three. Filed, not fixed, as the brief directed.

Nothing else was found that is not fixed on this branch.
