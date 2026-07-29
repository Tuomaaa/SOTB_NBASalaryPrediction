# Task brief: a waiver erases the price history — teach Stage 1 the interaction

Read `docs/worker-brief.md` first — own worktree, `OMP_NUM_THREADS=6`.

**Pin**: current `master` (v8.2x, A1 0.8210 / A2 0.8625 / B1 0.8528).
Deliver a branch + `docs/briefs/2026-07-29-waiver-interaction.RESULT.md`.

Two other workers are out (docs pass 2; two candidate features from the
floor-crash review). **Do not rebase onto either.** The two-features worker
adds columns through `scripts/build_external_features.py`; you must emit your
column from `scripts/build_waiver_features.py` instead, so the only file you
share with anyone is `FEATURE_COLS` in `src/model/train.py` — the architect
merges that line.

## The finding this exists to act on

`is_waived` landed in v8.2x at ΔSel +0.00666, t = 2.29. It works, but the
architect session then measured **where** it works, and the answer is narrow.
All numbers below are from the v8.2x champion's own OOF column on the 944-row
evaluation frame; reproduce them as your first step.

**A waiver flattens the relationship between last year's pay and this year's:**

| | OLS slope, this-year cap_pct on prev_cap_pct | Spearman |
|---|---|---|
| not waived (n=822) | **+0.750** | +0.594 |
| waived (n=122) | **+0.140** | +0.429 |

Landing spot by prior-pay band, in $M at the 2026 cap:

| prior pay | waived n | waived lands at | not-waived n | not-waived lands at |
|---|---:|---:|---:|---:|
| <3% of cap | 66 | $3.49M | 355 | $6.08M |
| 3–7% | 21 | $3.23M | 199 | $14.40M |
| 7–12% | 15 | $5.38M | 130 | $17.59M |
| 12–20% | 9 | $5.85M | 81 | $21.50M |
| **≥20%** | **11** | **$9.27M** | 57 | **$39.90M** |

Prior pay is near-identical between the two groups in aggregate (0.062 vs
0.066 of cap), so this is not "waived players were always cheaper".

**And the champion's error on waived rows is entirely in the top band:**

| prior pay | n | actual | predicted | bias |
|---|---:|---:|---:|---:|
| <3% | 66 | $2.71M | $2.93M | +0.21 |
| 3–7% | 21 | $2.65M | $3.36M | +0.70 |
| 7–12% | 15 | $3.97M | $3.43M | −0.54 |
| 12–20% | 9 | $4.21M | $4.67M | +0.46 |
| **≥20%** | **11** | **$7.67M** | **$13.11M** | **+5.44** |

Flat, flat, flat, flat, then +$5.44M. The model prices waived fringe players
correctly and over-prices waived stars, because it is still paying for a price
history the market has stopped paying for. Lillard 2025 is the extreme case:
`is_waived = 1` fired, and the prediction moved only $40.77M → $38.42M against
an actual of $14.10M.

`is_waived` is a single binary column, so a tree can only apply one average
discount to it — measured at **$1.59M mean, $0.94M median** across the 122
rows by an in-sample flip test. The discount the top band needs is an
interaction, and 11 rows out of 944 is not enough for a tree to find one
unaided.

## Part 0 — measure the headroom before you build (report this first)

The architect deferred this measurement to you rather than gating on it, so it
is the first thing in your RESULT and it stands on its own even if every arm
below fails.

1. What share of the frame's total squared error do those 11 rows carry?
   Lillard alone is ~3.2%; report the other ten individually and as a group.
2. An oracle bound: replace the champion's prediction on the 11 rows with the
   band's observed mean and recompute A1. That is the ceiling any interaction
   term can reach on this segment, and if it is under ~+0.004 say so plainly —
   a term that cannot clear the gate is still worth reporting as a measured
   dead end.
3. Both numbers with and without Lillard, since one row should not be the
   whole case.

## Part 0b — one row needs verifying first

**Bradley Beal, season 2025**, sits in the training frame at `salary_m`
**$19.383M**, `signing_cat` MLE, with `prior_waiver_text` = *"Waived by
Phoenix (PHX) via Buyout and Stretch Provision"*. He is the **only** one of
122 waived rows above 10% of the cap, and he is in your top band.

Beal was bought out by Phoenix in July 2025 and signed with the Clippers for a
reported 2 years / $11M. **$19.4M is approximately Phoenix's stretched dead-money
obligation** (~$97M over five years), not a contract Beal signed. If that is
what the row captures, it is a fabricated observation and it is sitting in the
exact band your feature targets.

Establish which it is from the cached salary and Spotrac sources. **Do not fix
it** — file it in `ISSUES.md` (next number is 32) with a reproducer, report both
sets of numbers (row in / row out) for every arm, and let the architect decide.
Then scan the frame for other rows whose salary matches a stretched obligation
rather than a signing; Lillard 2025 is correct ($14.104M = the Portland deal),
so this is not known to be systematic.

## Part 1 — Arm A, the registered form

Add ONE column, as the sixteenth feature:

```
prev_cap_pct_x_waived = prev_cap_pct * is_waived
```

Nothing else changes. The form is read directly off the two fitted lines above
(waived slope 0.140 vs non-waived 0.750) and takes **no tuned constant** — the
tree learns the discount. That is deliberate: a shrinkage form like
`prev_cap_pct * (1 - k * is_waived)` would require fitting `k` against the
target, and this project has killed three experiments on exactly that kind of
selection pressure.

`is_waived` is NaN where the source does not cover the row. Decide the NaN
semantics of the product **before** you score anything, state the decision, and
test only that form. Native NaN is the presumption (a NaN × a number is NaN,
and "we do not know whether he was waived" is genuinely unknown), but say so
explicitly — a native-NaN vs median-fill difference has flipped a verdict here.

## Part 2 — Arm B, one extension, reported regardless of outcome

`mpg` showed the largest attenuation of any feature between the two groups
(Spearman 0.776 → 0.320). Arm B is Arm A plus:

```
mpg_x_waived = mpg * is_waived
```

**Read the attenuation table with the caveat the architect attached to it**:
the waived group's target has much smaller spread (sd 0.0196 vs 0.0850) with
heavy ties at the minimum, and rank correlation attenuates mechanically under
that compression. So the across-the-board feature attenuation is *suggestive*
and Arm B is speculative. The prior-pay result in Part 0 does not depend on it:
the compression **is** the claim there, not a confound for it.

Run Arm A and Arm B, report both, and do not try a third form.

## Judging

Full challenger gates, each arm independently, per `docs/worker-brief.md`:

1. paired ΔSel ≥ +0.002 with **t > 2**, per-fold pairing (the fold is the unit;
   a seed-paired t inflates ~3×);
2. A2 moves the same direction;
3. B1 forward drop ≤ 0.003;
4. C2 fixed-segment |bias| growth ≤ $0.30M;
5. relative C1: |slope − 1| excess ≤ 0.005 vs the incumbent's 0.9709;
6. coverage-indicator control — `is_waived_known` already exists for this and
   the v8.2x RESULT ran it at +0.00029, t = 1.02. Your product inherits the
   same coverage structure, so run it again as an increment, not by assumption.

SHIP-FORM testing is mandatory: the column you score must be byte-for-byte
what `build_waiver_features` would emit.

## Segments to report for every arm, on fixed rows

- the five prior-pay bands above, champion and arm, bias and MAE each;
- the 11 top-band rows named individually, before and after;
- Lillard 2025 specifically;
- the 122 waived rows as a group, and the 822 non-waived rows as a group —
  **a term that only fires on 122 rows must not move the other 822**, and if it
  does, that is the finding;
- the standard C2 mechanism table.

## Traps

- **Do not build this as a Stage-2 bound.** The architect proposed exactly that
  ("waived ⇒ capped at the mid-level"; 0 of 122 rows exceed 15% of cap; an
  in-sample clip at that line reads +0.0078) and the user rejected it, correctly.
  Stage 2 holds *deterministic CBA bounds on the player himself*. No CBA
  provision caps a waived player's next contract; the regularity is empirical,
  it would be a one-way ratchet that permanently forbids pricing a bought-out
  star correctly, the 15% line is fitted to a 122-row sample maximum, and the
  standing decision that the Stage-2 clip never reads observed salary exists to
  keep that door shut. If your results tempt you back toward a clip, write the
  temptation into the RESULT and stop — do not implement it.
- **11 rows.** An interaction whose evidence is 11 rows is fragile by
  construction. Report the per-fold deltas in full, and say how many of the 11
  land in each fold; if one fold carries the whole gain, that is a red flag and
  belongs in your Anomalies section, not in the headline.
- Do not touch the filter chain, the ceilings, `stages.py`, or the route
  machinery. Features only.
- Do not touch `ISSUES.md` numbering conventions — numbers are permanent, never
  reused, next is 32.
- A delta above +0.01 is a re-verify flag, not a victory lap.

## Deliverable

Branch + RESULT: Part 0's headroom numbers first, the Beal verdict, the NaN
decision you registered, per-arm paired tables with per-fold values, the segment
tables above, the ship-form statement, and a plain adopt / do-not-adopt for each
arm. Expected effort: half a day to a day.
