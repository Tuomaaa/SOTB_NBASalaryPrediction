# RESULT: MLE mass-point reconnaissance (measure only)

**Task**: `docs/briefs/2026-07-24-mle-mass-point.md` · **Pin**: `master` 7718b27
(v7.9x line) · **Branch**: `worker/mle-mass-point` · **Type**: measurement only —
no table, filter, model, or version number changed.

## 1. What we found and why it matters

The CBA's mid-level/bi-annual exceptions are fixed dollar amounts per season, and
signings pile up **on those exact numbers**. Two independent instruments agree to
the dollar: the empirical spike-hunt of the year-1 salary distribution (the same
mass-point recovery `floor_pct` uses for the minimum) lands on the published
Hoops Rumors amounts for every (season, exception) that has any signing. The
pile-up is real and reproducible across all eight seasons and all four exception
types, and it survives the 2023-CBA redefinition (which flipped the taxpayer and
room ratios — see the ratio table).

The pile-up is **moderately thick and currently mispriced**. 67 rows sit exactly
on an exception amount (≤0.5%), 93 within ≤2% — that exact count already exceeds
the 57-row max-contract zone that earned its own censoring branch. The champion
**overprices** these rows by **+$0.83M** (driven by taxpayer MLE +$1.49M and BAE
+$1.44M). The oracle bound — snapping those rows to the exact amount — is worth
**+0.0143 pooled R²** and **−$0.69M** on the \$3-15M band, the largest single
structural opportunity measured on this frame recently. But the prize is an
*oracle*: three structural facts (below) cap the realizable gain well under it, so
the recommendation is "worth designing, conditioned on the mixture design landing
and on one more separability measurement" — not "adopt".

## 2. The exception amounts, per season, with sources

Curated from Hoops Rumors' annual "Values Of …/… Mid-Level, Bi-Annual Exceptions"
posts into `data/raw/raw_external/mle_exception_amounts.csv` (32 rows, one
`source_url` per amount). First-year starting salary in \$M; **cap_pct** in
parentheses uses `config.CAP_BY_SEASON`.

| season | non-tax MLE | taxpayer MLE | room MLE | BAE |
|-------:|:-----------:|:------------:|:--------:|:---:|
| 2019 | 9.258 (8.48%) | 5.718 (5.24%) | 4.767 (4.37%) | 3.623 (3.32%) |
| 2020 | 9.258 (8.48%) | 5.718 (5.24%) | 4.767 (4.37%) | 3.623 (3.32%) |
| 2021 | 9.536 (8.48%) | 5.890 (5.24%) | 4.910 (4.37%) | 3.732 (3.32%) |
| 2022 | 10.490 (8.48%) | 6.479 (5.24%) | 5.401 (4.37%) | 4.105 (3.32%) |
| 2023 | 12.405 (9.12%) | 5.000 (3.68%) | 7.723 (5.68%) | 4.516 (3.32%) |
| 2024 | 12.822 (9.12%) | 5.168 (3.68%) | 7.983 (5.68%) | 4.668 (3.32%) |
| 2025 | 14.104 (9.12%) | 5.685 (3.68%) | 8.781 (5.68%) | 5.134 (3.32%) |
| 2026 | 15.044 (9.06%) | 6.064 (3.65%) | 9.366 (5.64%) | 5.477 (3.30%) |

**The 2023 CBA redefinition is real and must not be extrapolated across** (brief
trap #1): non-tax MLE 8.48% → 9.12% of cap, taxpayer MLE **5.24% → 3.68%** (a
drop, plus a 3→2 year cap), room MLE **4.37% → 5.68%** (a jump). BAE alone holds
at 3.32%. A single ratio-to-cap projected across the boundary would misplace three
of the four amounts.

**Empirical ↔ published agreement.** Every published amount that has ≥1 signing in
the frame has empirical mass at ≤0.5% relative distance — e.g. 2019 room \$4.767M
(×10 exact), 2020 non-tax \$9.258M (×6) and BAE \$3.623M (×3), 2022 taxpayer
\$6.479M (×4), 2023 taxpayer \$5.000M (×5), 2024 room \$7.983M (×2). Only 2022 BAE
and 2022 room have **zero** frame rows — thin, not contradictory (no instrument
disagreement; two instruments simply have no observation there). The `is2019`-style
season-alignment trap does not apply — amounts are CBA constants, not collection
artifacts.

## 3. Pile-up thickness (949 evaluation frame)

Each row assigned to its **nearest** exception by relative salary distance
(instrument = salary proximity; label used only as cross-check, per brief).
"±1%/±2%" read as relative to the exception amount (a ±2% band on a \$10M MLE is
±\$200K — the right scale for a mass point; ±2% *of cap* would be an absurd ±\$3M).

| tolerance | rows | % of 949 |
|:--|--:|--:|
| exact (≤0.5%) | 67 | 7.1% |
| ≤1% | 75 | 7.9% |
| ≤2% | 93 | 9.8% |

Within the \$3-15M band the concentration is **90 / 367 = 24.5%**. Split across
types (≤2%): non-tax MLE 28, room 25, taxpayer 25, BAE 15 — all four carry mass,
all eight seasons represented. Per-season × type and per-`signing_cat` tables are
in the run log (`scratchpad/mle_analysis.py`). Multi-year MLE escalation is a
non-issue: the frame is year-1 only, and year-2+ escalated rows were confirmed
absent from the mass points (brief trap #2).

Label composition of the 93 near-exception rows: **MLE 50**, Unknown 14, Other 10,
Bird Rights 10, Cap Space 3, Early Bird 3, Non-Bird 2, Sign & Trade 1. Two facts
this exposes:
- Only **50 / 124 (40%)** of MLE-*labeled* rows sit within ≤2% of a full exception
  amount. The other 60% signed for **less than** the full exception (partial-MLE)
  and are spread across the band — so the mass-point structure captures under half
  of the MLE mechanism even with a perfect label.
- 43 / 93 (46%) near-exception rows are **not** MLE-labeled (§5).

## 4. Does the champion already price them? (frozen v7.9x OOF)

Champion out-of-fold from `outputs/models/oof_reference.csv` (the 7718b27
champion; pooled R² reproduced at **0.7631**, and the MLE C2 slice reproduced at
n=124 / bias +\$0.833M / MAE \$2.721M — matches `evaluation_suite.json` to the
digit, so the OOF is the right reference).

| group | n | bias (\$M) | MAE (\$M) |
|:--|--:|--:|--:|
| \$3-15M band (all) | 367 | −0.060 | 3.012 |
| — at exception (≤2%) | 90 | **+0.834** | 2.853 |
| — rest of band | 277 | −0.351 | 3.063 |
| — non-tax MLE | 25 | −0.179 | 4.256 |
| — taxpayer MLE | 25 | **+1.485** | 2.338 |
| — room MLE | 25 | +0.831 | 2.562 |
| — BAE | 15 | **+1.442** | 1.858 |

The model systematically **overprices** exception signings (it reads their box
production as worth more than the exception pays). Taxpayer-MLE and BAE deals —
good players a capped-out contender adds cheaply — are the worst, +\$1.4-1.5M.

## 5. Oracle bound and the false-positive surface

**Oracle** (mirror of the max-zone oracle): set prediction = the exact exception
cap_pct for the 90 at-exception (≤2%) band rows, leave all others at champion OOF.

| metric | champion | oracle | Δ |
|:--|--:|--:|--:|
| \$3-15M band MAE | 3.012 | 2.319 | **−0.693M** |
| pooled R² (949) | 0.7631 | 0.7774 | **+0.0143** |
| at-exception zone MAE | 2.853 | 0.027 | −2.826M |
| at-exception zone bias | +0.834 | −0.003 | — |
| calibration slope | 0.9881 | 1.0028 | toward 1 |

That +0.0143 is above the worker brief's +0.01 red-flag line **by construction and
expectedly**: an oracle snaps rows to the answer using knowledge of the observed
salary. It is the *ceiling*, not a result — the honest interpretation is "a perfect
mass-point branch could recover at most this much."

**False-positive surface.** Of the 93 near-exception rows, **43 (46%) are
coincidental non-MLE landers** — BAE-band "Other" (9), non-tax-band Unknown (7) and
Bird Rights (6), etc. A Bird re-signing at ~\$9.5M has nothing to do with the MLE;
it just lands there. The snap *error* the oracle imposes on these is tiny (mean
\$0.047M, bounded by 2% of the amount) because they are already near the amount —
**but that understates deployment risk**: the oracle fires on known-at-amount rows,
whereas a real branch must fire on *features*, which cannot separate an MLE signing
from a coincidental \$9.5M Bird deal (the deciding variable is the signing team's
cap posture — not a feature, and "team cap space" is already a rejected candidate
per CLAUDE.md). The real exposure is therefore the 277 rest-of-band rows a
features-only classifier would wrongly pin, not the 43.

## 6. Recommendation

**Worth designing a third mass-point branch — conditional, and with a bounded
prize — not a standalone adopt.** The pile-up clears the thickness bar (67 exact
> the 57-row max zone that earned a branch), it is genuinely mispriced (+\$0.83M,
up to +\$1.5M on taxpayer MLE / BAE), and the oracle ceiling (+0.0143 R², −\$0.69M
band MAE, calibration slope 0.9881→1.0028) is the largest structural prize on this
frame in recent memory. So **if** the mixture-output design (#4/#8) lands, adding a
season-indexed mass point at the four exception cap_pcts is cheap and points the
right direction. **But** three facts cap the realizable gain far below the oracle,
and any adopter must respect them: (1) the MLE is a **mid-distribution attractor,
not a bound** — there is no "the ceiling binds" corroboration signal the way max
(`is_max_contract`) and floor (`is_at_floor`) branches have, so the branch cannot
lean on the model to confirm membership; (2) membership is set by the team's cap
posture, which is **absent from the 14 features** (and team-cap-space is a rejected
candidate), so a features-only branch recovers only the separable fraction; (3)
even a *perfect* mechanism label pins only 40% of MLE rows (the rest are
partial-MLE), and 46% of proximity hits are coincidental. **Next step before
committing a branch: measure features-only separability of at-MLE rows** (a cheap
classifier AUC on the \$3-15M band) — that fraction, not the oracle, sets the true
prize. Question narrowed, not closed.

## 7. Anomalies

- **+0.0143 oracle > +0.01 red-flag line** — expected: an oracle uses the observed
  salary to snap. Mechanism stated in one sentence (§5); not a champion result.
- **`config.py` 2026 cap is stale, and `check_caps` can't see it** — official
  2026-27 cap is \$164,961,000 ([pr.nba.com](https://pr.nba.com/2026-27-salary-cap/));
  config carries 166,000,000 (+0.63%). `check_caps.py` reports 2026 "ok" anyway,
  because five 2026 extension-max salaries are *also* stale-projected against
  166.0M (\$49.8M = 0.30×, \$41.5M = 0.25×), so cap and data agree in lockstep
  while two real FA signings (Reaves, Young) already sit on the official cap.
  Fixing the cap alone would flip `check_caps` to "WRONG". Filed as **ISSUES #16**
  (architect/data lane — must fix cap and the five salaries together). Nudges my
  2026 cap_pct figures ≤0.63%, immaterial to the conclusions (2026 holds few
  at-exception rows).
- **2022 has zero frame rows at BAE and room MLE** — thin, not an instrument
  disagreement (both routes simply lack an observation there).

## 8. Files, reproduction, gate verdicts

**Touched (this branch only):**
- `data/raw/raw_external/mle_exception_amounts.csv` — new curated amounts (deliverable artifact)
- `docs/briefs/2026-07-24-mle-mass-point.RESULT.md` — this report
- `ISSUES.md` — appended #16 (stale 2026 cap)

**Experiment scripts** (scratchpad, not committed):
`scratchpad/spike_hunt.py`, `scratchpad/mle_analysis.py`. Reproduce from the
worktree root with `OMP_NUM_THREADS=6 python scratchpad/mle_analysis.py`; it loads
the 949 frame via `load_evaluation_frame()` and the frozen champion OOF from
`outputs/models/oof_reference.csv`.

**Gate verdicts**: none apply — this is a measurement task. No paired ΔSel, no
model/feature/filter change, no accept/reject. The confirmation split was not read
into any statistic here.

**Proposed commit message** (architect edits/lands; no version number — nothing moved):

```
Recon: MLE mass-point measurement + curated exception amounts (no model change)

Curates the four CBA exception amounts (non-tax/taxpayer/room MLE, BAE) for
2019-2026 from Hoops Rumors into data/raw/raw_external/mle_exception_amounts.csv
with per-amount source URLs; empirical spike-hunt corroborates every amount to
<0.5%. On the 949 frame: 67 rows exact / 93 within 2% of an exception (24.5% of
the $3-15M band); champion overprices them +$0.83M (taxpayer MLE +$1.49M, BAE
+$1.44M). Oracle ceiling for a snap/branch: +0.0143 pooled R2, -$0.69M band MAE,
calibration 0.9881->1.0028. Recommendation: third mixture branch worth designing
IF the mixture design lands, prize bounded well under the oracle by MLE
membership being a team-cap (non-feature) attractor, not a corroborable bound.
Files ISSUES #16 (config 2026 cap 166.0M stale vs official 164.961M).
```
