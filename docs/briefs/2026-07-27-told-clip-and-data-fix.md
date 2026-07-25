# Task brief: fix ISSUES #22/#23, then re-measure the told-route Stage-3 clip

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`. A second worker is running the floor-side experiment
(`2026-07-27-floor-branch.md`); it touches `route_mixture.py` and its own
harness, you touch the salary tables and `extension_cap.py`. Do not pull
mid-task.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-27-told-clip-and-data-fix.RESULT.md`.

## Context: the architecture decision this serves

The three-stage architecture is settled:

1. **Stage 1** — the player's value, unconstrained.
2. **Stage 2** — clip into the CBA bounds that apply to the player himself
   (his max tier, the league minimum). Deterministic, no route needed.
3. **Stage 3** — adjust for the **signing route once it is known**: an
   extension is capped at its raise limit, an MLE signing snaps to the
   exception amount, a Bird re-signing carries its premium.

**Decision, 2026-07-26 (user's, overriding an earlier position of mine):**
the Stage-3 number is reported as the model's accuracy in the SAME column as
the ex-ante numbers, not a separate scoreboard. The route is information that
is available at prediction time for both deployed uses — pricing an unsigned
free agent (who by definition is not extending) and analysing a signed
contract (whose route is known). The champion has the same information and
simply ignores it, so this is "uses an available covariate" and not leakage.
**One bookkeeping requirement**: v7.1x–v7.13x were all computed under the
"ignore the route" convention, so any RESULT and any version entry must state
plainly that the convention changed here, so the old numbers are not compared
naively against the new ones.

## What the architect already measured (reproduce it, then improve it)

`push_then_told_clip` at the pre-registered τ = 0.52: push toward the tier
ceiling where P(max) ≥ τ, then clip at the row's OWN extension cap for rows
that ARE first-paying-year extensions (`is_extension`, `ext_cap_pct`).

| arm | A1 | MAE | max-zone MAE | 25%+ band bias |
|---|---|---|---|---|
| champion | 0.7865 | $3.076M | $4.297M | +1.06 |
| push only | 0.7849 | $3.026M | **$3.179M** | **+2.27 (brake fails)** |
| **push + told clip** | **0.7936** | **$2.945M** | **$3.179M** | **+1.13 (brake passes)** |

The clip keeps the whole zone win and removes the brake failure that killed
the branch in phases 2 and 3: Aldridge 2019 +$12.20M → +$0.82M, Brunson 2025
+$11.45M → $0.00M. Paired ΔSel is +0.0037 at t = 0.43 — the effect is real in
the pooled numbers but fold-noisy, because it moves only 12 rows and **the two
largest remaining errors are both known data defects**, which is what Part 1
fixes.

## Part 1 — the data fixes

**ISSUES #23 — Marcus Smart 2022.** He is granted a designated-veteran
exemption from an award he won AFTER signing his extension (the mirror of the
#19 bug). Because he reads exempt, his extension cap is the tier $43.28M
instead of his real $16.61M, and he keeps **+$22.22M of error** in the arm
above — the single largest row in the whole experiment. Fix: test
designated-veteran eligibility against awards that PRECEDE the signing date,
which `contract_signing_dates.csv` now provides. Note the worker's own warning
in #23: fixing this moves Smart from "exempt" to "over-cap" until #22 lands,
so land them together and check the hard gate at the end, not in between.

**ISSUES #22 — three over-cap rows.** All three are defects in our salary
table, each already diagnosed to the dollar:

- *Dejounte Murray 2024* — a trade bonus is added to the observed pay; the
  negotiated first year is exactly 140% of $18.214M. Decide and state whether
  the fix is to record the bonus separately or to exclude such rows from the
  cap gate; do not clamp the ceiling to observed pay.
- *Ivica Zubac 2025 and Aaron Gordon 2026* — renegotiate-and-extend deals
  whose renegotiated prior-season salary our table does not carry.
  `1.4 × $13,495,700` is Zubac's pay to the dollar and
  `1.4 × $24,041,455` is Gordon's; Gordon's row carries an independent tell
  (his 2024 and 2025 both read $22,841,455, a duplicate). The repair path in
  #22 needs no new fetch.

**Gate**: after both fixes, over-cap rows among the 156 extension rows return
to ~0 at the 1e-4 tolerance, and no row anywhere is paid above its own
ceiling.

## Part 2 — re-measure

Re-run the arm above on the repaired data, at the same pre-registered τ = 0.52
(do not re-tune it). Report:

- the three-row table (champion / push only / push + told clip) with A1, MAE,
  max-zone MAE, the counterweight and 25%+ band brakes, C2, B1 forward, and
  paired ΔSel with its per-fold values;
- the per-row table for Smart, Aldridge, Brunson, Murray, Zubac and Gordon,
  before and after;
- **how much of the improvement is Smart alone** — the architect's estimate is
  that fixing #23 is worth about +0.0045 of A1 by itself, and a claim that
  size deserves its own line.

Then state plainly whether the arm clears the gates on the repaired data:
paired ΔSel t > 2, both brakes ≤ +$0.30M, C2 ≤ $0.30M, B1 drop ≤ 0.003. If
t is still short of 2 while A1 and MAE both improve, say so and leave the
adoption call to the architect — do not argue for a gate change.

## Traps

- The told clip reads `is_extension`, a realized-route fact. That is the
  point (Stage 3 is the told mode) but it must be stated in the RESULT, and
  the ex-ante variant (clip by P(extension) ≥ 0.50) must be reported beside
  it so the two are never confused.
- Do not clamp any ceiling to observed pay — that is the standing decision in
  `docs/QUEUE.md`, and the over-cap count is our most sensitive detector of
  exactly the data defects Part 1 is fixing.
- `ext_cap_pct` still must not enter `max_eligible_pct` or the Stage-1 censor
  mask.
- τ = 0.52 is fixed. Re-selecting it after seeing the repaired numbers is the
  selection pressure three experiments have died on.

## Deliverable

Branch + RESULT: the two fixes with their verification, the re-measured arm
table, the per-row before/after, the Smart-alone decomposition, the gate
verdicts, and the convention note. Expected effort: half a day to a day.
