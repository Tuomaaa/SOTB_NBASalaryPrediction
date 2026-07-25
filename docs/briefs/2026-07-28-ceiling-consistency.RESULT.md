# RESULT — ISSUES #27 and #26: ceiling consistency

**Branch**: `worker/ceiling-consistency` (worktree `../BBall-worker-ceiling-consistency`),
based on `master` @ `9813a0f`, the tag the brief pinned.

| commit | what |
|---|---|
| `7e61740` | ISSUES #27 — signing-season award anchor in `_compute_max_eligible` |
| `0429f8f` | ISSUES #26 — stage-3 rebuild of the stored `prev_cap_pct` |
| `d9e3094` | this RESULT |

`master` advanced to `7c317e9` (the docs pass and the injury-probe dispatch)
while this ran. Those two commits touch `VERSION_HISTORY.md`, `docs/QUEUE.md`
and other brief files only, so there is no overlap with the three files below
and the merge is clean.

---

## 1. What changed and why

**#27, route 2 (the real fix).** `_compute_max_eligible` had no signing
instrument, so its supermax branch judged Designated-Veteran eligibility off
awards the paying season could see — which is how Marcus Smart 2022 kept a 35%
tier ceiling ($43.28M) granted by a DPOY announced nine months *after* his
August-2021 extension was signed. A new `_signing_seasons()` reuses
`parse_signing_dates.contract_spans`, the same offline, cache-only instrument
`extension_cap` already uses; the supermax award test now reads the signing
season, and an undated row falls back to the paying season, reproducing the
pre-instrument answer exactly (197 of 1,297 rows take that fallback).

Two things did **not** move, and both were load-bearing. The **Rose Rule** stays
on the paying anchor: its 30% escalator is a conditional term written *into* a
rookie-scale extension, so the qualifying award legitimately postdates the
signature — moving it with the supermax test stripped the escalator from
thirteen rows genuinely paid at it (Trae Young 2022, Haliburton 2024, Edwards
2024, Mobley and Cunningham 2025 among them), and nine of the thirteen land
above their own tier ceiling as a result.
The **service window** stays on the paying season, per the brief, because what
is unreliable there is our debut-based count and not the calendar. A new
`DVE_FIRST_SIGNING_SEASON = 2017` guard is what makes that safe: the Designated
Veteran exception was created by the 2017 CBA, and without the guard a
paying-season service count of 7–9 paired with a pre-2017 award promotes plain
pre-DVE maxes (§2 has the audit).

**#26.** `salary_corrections.csv` is applied inside `_load_prev_season_cap_pct`,
which the ceiling rules read at load time, but the `prev_cap_pct` **feature** was
baked into `training_data_v2.csv` before the corrections existed. Stage 3 was
re-run in isolation, per the prev-cap/feature-batch precedent, and verified cell
by cell: `prev_cap_pct` is the only column that moves, on exactly the three rows
ISSUES #26 names.

---

## 2. The supermax audit (route 2)

`_compute_max_eligible` sees 1,297 rows. **Four** ceilings change; the frame that
reaches training is 944 and **one** of the four is in it.

| player | season | signed | exp | pay | tier ref → new | ceiling ref → new | in the 944 frame? |
|---|---|---|---|---|---|---|---|
| marcus smart | 2022 | 2021 | 8 | $17.46M | 0.35 → **0.30** | $43.28M → $37.10M | **yes** |
| blake griffin | 2019 | 2017 | 9 | $34.23M | 0.35 → 0.30 | $38.20M → $37.13M | no |
| damian lillard | 2019 | 2015 | 7 | $29.80M | 0.35 → 0.30 | $38.20M → $32.74M | no |
| anthony davis | 2019 | 2015 | 7 | $27.09M | 0.35 → 0.30 | $38.20M → $32.74M | no |

Row by row: Smart is ISSUES #27 itself. Griffin's covering deal is the July-2017
LAC re-signing, a 30% max (his 2017 pay is 29.8% of that cap) — the All-NBA that
fired the old rule is from the 2018-19 season, two years after the ink. Lillard's
is the July-2015 max and Davis's the July-2015 max, both signed before the
Designated Veteran exception existed at all. All four demotions are downward and
none puts a row above its own ceiling.

**The 14 designated-veteran rows the extension work identified**: 13 are
unchanged, and the fourteenth is Smart.

| unchanged | Lillard 2021, Booker 2024, Mitchell 2025, Giannis 2021, Harden 2019, Brown 2024, Tatum 2025, Embiid 2023, Wall 2019, Randle 2022, Towns 2024, Jokic 2023, Gobert 2021 |
|---|---|
| **changed** | **Smart 2022** (0.35 → 0.30) |

Both asymmetries the extension worker documented survive. Wall 2019 and Harden
2019 (signed two summers early) keep their ceilings because their qualifying
awards precede the signing too; Embiid 2023 keeps his 35% because the
paying-season service anchor rescues a debut-based count that reads 5 where the
CBA saw 7.

### The variant I rejected, and why

Taking "keep the service anchor unioned" to mean *also* qualifying on the
signing-season service count is what the brief literally asks for. I ran it: it
gives **identical ceilings** to the shipped rule (its four extra supermax flags
all land on `exp >= 10` rows, where the base tier is already 0.35), so nothing
turns on the choice. I also ran a fully self-consistent variant (award and
service both read at the same anchor, unioned over anchors) and **rejected** it —
it demotes Embiid 2023 to a 30% tier when he is paid exactly 35%, which pushes
him over his own ceiling, drops him at the mislabel filter, and takes the frame
to **943**. Changing the row count is an escalate-first condition, so that
variant is out.

The one row I could not justify is what the pre-2017 guard exists for. Without
it, **Andre Drummond 2019** is *promoted* 0.30 → 0.35: his covering deal is the
July-2016 DET re-signing, where the paying season's service count (7) pairs with
a 2016 All-NBA that precedes the signing. His signing-season service count is 4,
the deal predates the DVE by a year, and it is a plain max. Rather than add a
curated `designated_ineligible` row for a case the rule gets wrong — and one that
never reaches the frame anyway — the guard states the legal fact that no
pre-2017 signing can be a Designated Veteran deal. It is the same class of
statement as the 25/30/35 tiers, not another line on a list that needs a new
entry every summer (ISSUES #6).

**Verifications the brief asked for**: Smart 2022 reads `tier_ceiling_pct ==
0.30`; over-ceiling count is **0 → 0**; the frame is **944 → 944**;
`is_max_contract` is **70 → 70** (Smart's $17.46M is 47% of even the corrected
ceiling, so his flag was never in play).

---

## 3. The #26 rebuild: what moved

Stage 3 alone (`phase3.build_contract_features`), not the full chain. Verified
cell by cell over all **3,113 rows × 36 columns**:

```
columns that moved: ['prev_cap_pct']        3 cells
```

| player | season | prior season | stored | rebuilt |
|---|---|---|---|---|
| marcus smart | 2022 | 2021 | $13.84M | **$14.34M** |
| ivica zubac | 2025 | 2024 | $11.74M | **$13.50M** |
| aaron gordon | 2026 | 2025 | $22.84M | **$24.04M** |

Exactly the three rows and three figures ISSUES #26 names, and all three now
agree with `_load_prev_season_cap_pct()` to within 1e-12. Row count 3,113
unchanged, row keys unchanged.

Two deviations from `rebuild_training_data.stage3`, both deliberate and both
worth an ISSUES entry (§7):

- **`is_contract_year` preserved, not dropped.** `stage3` drops the column, but
  the stored file still carries it and nothing outside `phase3` reads it.
  Dropping it is a schema change outside this task's scope, so it was kept — and
  compared: it recomputes **identically**, 0 disagreeing cells.
- **Written as a textual field edit.** A pandas round-trip re-serialises the
  whole `prev_cap_pct` column at a different precision
  (`0.0139814366899228` → `0.013981436689922879`) and `phase3`'s sort reorders
  2,259 of 3,113 lines. Either alone turns a 3-cell change into a whole-file
  diff. The values come from `build_contract_features`; only their placement is
  textual, so the other 3,110 rows stay byte-identical.

---

## 4. Numbers

Champion Grabit fitter, 10 seeds × 5 GroupKFold folds. All four arms score the
**same 944 rows in the same order**, asserted before pairing, so the folds are
identical and the per-fold pairing is valid. `ref` = master code + master data.

| arm | n | A1 | A2 | B1 | A1 MAE | max zone n / MAE | is_max | over-ceiling | C1 slope |
|---|---|---|---|---|---|---|---|---|---|
| ref | 944 | 0.78653 | 0.83412 | 0.83417 | $3.076M | 70 / $4.297M | 70 | 0 | 0.98086 |
| fix27 only | 944 | 0.78653 | 0.83412 | 0.83417 | $3.076M | 70 / $4.297M | 70 | 0 | 0.98086 |
| fix26 only | 944 | 0.78668 | 0.83393 | 0.83412 | $3.075M | 70 / $4.293M | 70 | 0 | 0.98103 |
| **both** | 944 | **0.78668** | **0.83393** | **0.83412** | $3.075M | 70 / $4.293M | 70 | 0 | 0.98103 |

**Paired ΔSel vs ref, per-fold pairing (the fold is the unit):**

| arm | ΔSel | SE | t | per-fold |
|---|---|---|---|---|
| fix27 only | **+0.00000** | 0.00000 | — | `[0, 0, 0, 0, 0]` |
| fix26 only | +0.00020 | 0.00026 | +0.79 | `[+0.00016, +0.00013, −0.00026, −0.00019, +0.00118]` |
| **both** | **+0.00020** | 0.00026 | **+0.79** | `[+0.00016, +0.00013, −0.00026, −0.00019, +0.00118]` |

Pooled (context only, not the decision metric): both = +0.00025, t = +0.95.

**B1 per origin:**

| origin | n | ref R² | both R² | ref MAE | both MAE |
|---|---|---|---|---|---|
| 2024 | 112 | 0.87568 | 0.87564 | $2.993M | $2.996M |
| 2025 | 117 | 0.81732 | 0.81695 | $3.715M | $3.713M |
| 2026 | 95 | 0.79638 | 0.79679 | $3.511M | $3.506M |

B1 overall 0.83417 → 0.83412 (CI95 [0.7685, 0.8847] → [0.7685, 0.8850]).

**C2, fixed row groups taken from `ref`** — largest |bias| move in any segment is
**$0.0144M** (Sign & Trade, n=13), against a $0.3M budget. Every other segment
moves less than $0.006M.

**The three touched rows** (out-of-fold prediction, ref → both):

| row | actual | ref | both |
|---|---|---|---|
| marcus smart 2022 | $17.46M | $28.60M | $28.62M |
| ivica zubac 2025 | $18.89M | $39.26M | $39.53M |
| aaron gordon 2026 | $33.66M | $19.60M | $19.87M |

**Verdict against the brief's rule** ("adopt unless the paired delta is
significantly negative, t < −2"): t = **+0.79**. Adopt.

---

## 5. Anomalies

**The brief's premise that #27 moves the champion is wrong — it is a provable
no-op.** The `fix27` arm's out-of-fold predictions are **bit-identical** to
`ref` (max |diff| = 0.000e+00), which is why its paired delta is exactly zero
rather than merely small. The mechanism, in one sentence: no ceiling column is a
model feature (the 14 are `darko_dpm_z … prev_cap_pct`), so `max_eligible_pct`
reaches the champion only as the Stage-1 censor mask and the Stage-2 clip's upper
bound — and Smart is censored by neither (his `is_max_contract` is False at both
ceilings) and clipped by neither (his prediction is $28.60M against a $37.10M
ceiling). I flag this rather than bury it because the brief pre-committed to a
version bump: **#27 alone does not earn one**, and the whole branch's movement
comes from #26.

**Blake Griffin 2019 changes which filter drops him.** Lowering his tier to 0.30
puts his $34.23M above it, so the mislabel filter now catches him (6 → 7 rows)
and the continuation filter drops one fewer (347 → 346). The net frame is
unchanged at 944 and he is out either way, but the two filters' printed counts
move and a reviewer diffing logs will see it. This is arguably an improvement:
the mislabel filter is the more specific instrument and it now names him.

**No delta is anywhere near the +0.01 red-flag line.** The largest is +0.0002.

**Smart's extension ceiling is unaffected**, which matters for the merge:
`ext_cap_pct` for that row is $17.457M both before and after, because the raise
cap (1.20 × $14,339,285 + $250,000) binds far below either tier, so `min(tier, …)`
never selects the tier. `ext_is_dvp` was already False — `extension_cap` moved to
the signing anchor at ISSUES #23.

---

## 6. Files touched, and the merge boundary

| file | change |
|---|---|
| `src/model/train.py` | new `_signing_seasons()`, new `DVE_FIRST_SIGNING_SEASON`, supermax branch and docstring inside **`_compute_max_eligible`** |
| `data/processed/training_data_v2.csv` | 3 cells of `prev_cap_pct` |
| `docs/briefs/2026-07-28-ceiling-consistency.RESULT.md` | this file |

**For the architect merging against the stage-3 wiring worker**: in `train.py` I
touched **only** `_compute_max_eligible` and added two module-level definitions
immediately above it (`_signing_seasons`, `DVE_FIRST_SIGNING_SEASON`). I did not
touch the Stage-2/3 composition, `train_grabit`, the fitters, `extension_cap.py`,
`evaluate_suite.py`, `export_web.py` or `predict.py`. The one cross-lane fact
they need: **`tier_ceiling_pct` for Marcus Smart 2022 is now 0.30, not 0.35** —
`extension_cap.attach_extension_cap` reads that column, and I verified above that
his `ext_cap_pct` does not move as a result.

Experiment scripts stayed in the scratchpad, not the repo.

---

## 7. Proposed commit message

(The two commits on the branch are already written; if the architect squashes:)

```
Anchor the supermax ceiling on the signing season; refresh the stored prev_cap_pct

ISSUES #27: _compute_max_eligible had no signing instrument, so its supermax
branch read Marcus Smart 2022 at a 35% Designated-Veteran ceiling ($43.28M)
granted by a DPOY announced nine months after his August-2021 extension was
signed — the same wrong fact ISSUES #23 fixed in extension_cap. The award test
now reads the signing season via parse_signing_dates.contract_spans, the same
offline instrument extension_cap uses; undated rows fall back to the paying
season and reproduce the previous answer exactly. Rose Rule deliberately stays
on the paying anchor (its escalator is a contract term whose award postdates the
signature) and a pre-2017 guard keeps plain pre-DVE maxes off the award path.
Four ceilings move, all downward, all justified row by row; 13 of the 14 known
designated-veteran rows are untouched. Frame 944, is_max_contract 70,
over-ceiling 0, all unchanged.

ISSUES #26: the prev_cap_pct feature was baked into training_data_v2.csv before
salary_corrections.csv existed, so Smart 2022, Zubac 2025 and Gordon 2026
trained on a stale prior while their ceilings used the corrected one. Stage 3
re-run in isolation; verified cell by cell that prev_cap_pct is the only column
that moves, on those three rows only.

Paired dSel +0.00020 +/- 0.00026 (t = +0.79, per-fold pairing, 10 seeds x 5
folds, identical 944 rows). A1 0.78653 -> 0.78668, A2 0.83412 -> 0.83393,
B1 0.83417 -> 0.83412. Max zone 70 rows, MAE $4.297M -> $4.293M. C2 largest
fixed-segment |bias| move $0.014M. The ceiling fix alone is bit-identical on
predictions — no ceiling column is a feature and Smart is neither censored nor
clipped — so all movement comes from the prev_cap_pct refresh.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

## 8. ISSUES.md text (for the cleanup worker — I did not touch the file)

**Delete `## 26` and `## 27`.** Both are fixed on this branch and verified by
their own "Fixed when" criteria: Smart 2022 reads a 30% tier ceiling, and the
three `prev_cap_pct` rows agree with the loader with the frame still at 944.

**Add:**

```markdown
## 28. `rebuild_training_data.py` cannot produce a readable diff

**Severity**: low — no wrong number, but it hides wrong numbers.

Two independent sources of churn turn any stage-3 change into a whole-file diff:

- `phase3.build_contract_features` sorts by `(player_name_norm, season)` and the
  stored file is not in that order, reordering 2,259 of 3,113 lines;
- `prev_cap_pct` is stored at roughly 16 significant digits, so a pandas
  round-trip rewrites every row of the column
  (`0.0139814366899228` -> `0.013981436689922879`).

The 2026-07-28 ceiling-consistency rebuild changed 3 cells and produced a
2,259-line diff before it was written as a textual field edit instead. Anyone
reviewing a future refresh by eye will see nothing.

**Reproduce**: run `scripts/rebuild_training_data.py`'s `stage3_prev_cap_pct()`
alone on an unchanged tree and `git diff --numstat data/processed/training_data_v2.csv`.

**What to do**: have stage 3 write back into the incoming row order, and pin a
`float_format` for the file so the round-trip is stable. Do not reorder the
stored file to match the sort — that is the same 2,259-line diff, once.

**Fixed when**: re-running stage 3 with no input change produces an empty diff.

---

## 29. `is_contract_year` is a dead column the rebuild would silently drop

**Severity**: low.

`data/processed/training_data_v2.csv` carries `is_contract_year` (36 columns),
but `rebuild_training_data.stage3_prev_cap_pct` drops it before writing, so the
next full rebuild changes the file's schema as a side effect of an unrelated
refresh. Nothing outside `phase3.py` and `rebuild_training_data.py` reads the
column — it is not a feature and not in `STABLE_FEATURES`, so the rebuild's own
validator would not report the loss either.

The 2026-07-28 stage-3 rebuild preserved it deliberately and confirmed it
recomputes identically (0 of 3,113 cells disagree), so the choice is free: keep
it or drop it, but do it as its own decision.

**Reproduce**: `head -1 data/processed/training_data_v2.csv | tr ',' '\n' | grep -c is_contract_year`
returns 1; the column is absent from anything `rebuild_training_data.py` writes.

**Fixed when**: either the column is gone from the stored file and from
`build_contract_features`' output contract, or `stage3_prev_cap_pct` stops
dropping it.

---

## 30. `extension_cap._designated_veteran` mixes anchors the same way the ceiling path did

**Severity**: low today, latent — it cannot currently produce a wrong ceiling.

`_designated_veteran` loops over `(sign_season, season)` but passes
`award_anchor` (the signing season) to `_elite_trigger` on **both** iterations.
So on the paying-season pass it pairs a paying-season service count with a
signing-season award — the two tests describe different moments, and a deal
signed years before the paying season can satisfy both halves without ever
having been eligible.

This is harmless in `extension_cap` because a `True` verdict only means "the tier
ceiling governs instead of the raise cap", never an upgrade to 35%. It was NOT
harmless in `train._compute_max_eligible`, where the same shape promoted Andre
Drummond 2019 from a 30% to a 35% ceiling off his July-2016 re-signing — a deal
that predates the Designated Veteran exception. `train.DVE_FIRST_SIGNING_SEASON`
now blocks that class there; `extension_cap` has no equivalent guard.

**Reproduce**: call `_designated_veteran("andre drummond", 2019, 2016, ...)` with
the loaded elite/early/ineligible sets.

**What to do**: apply the same `signing_season >= 2017` guard in
`_designated_veteran`, or evaluate each anchor self-consistently there — but note
that the self-consistent form demotes Embiid 2023, whose debut-based service
count reads 5 where the CBA saw 7, so the guard is the cheaper of the two.

**Fixed when**: the two modules agree on what makes a row a Designated Veteran,
with Embiid 2023 still eligible and Drummond 2019 still not.
```
