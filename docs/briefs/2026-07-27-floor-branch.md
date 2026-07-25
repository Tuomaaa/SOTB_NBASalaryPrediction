# Task brief: the floor side — the largest unclaimed headroom in the project

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`. A second worker is fixing ISSUES #22/#23 and re-measuring
the max side (`2026-07-27-told-clip-and-data-fix.md`); its rows (Smart,
Murray, Zubac, Gordon) are not floor rows, so the two are independent. Do not
pull mid-task.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-27-floor-branch.RESULT.md`.

## Why this is being reopened

Phase 2 declared the floor branch NO-GO on an argument, not a measurement:
"the floor is already left-censored in Stage 1, so the rows a floor
classifier is confident about are the ones the left-censor already pins."
**The max side's version of that same argument was wrong**, and the架构 that
disproved it — a push plus a Stage-3 clip — has now been measured.

The architect measured the floor headroom on 2026-07-26 and it is the largest
single number in the project:

| zone | n | MAE | bias |
|---|---|---|---|
| max | 70 | $4.30M | **−$4.27M (under)** |
| **floor** | **240** | **$2.10M** | **+$2.04M (over)** |

**Oracle — pull every true at-floor row exactly onto its floor: A1 0.7865 →
0.8261, +0.0396**, MAE $3.076M → $2.575M. Six times the max side's oracle.

The reason the headroom exists is structural: **the Stage-2 clip only pushes
UP**. A player the model prices at $20M who signed for $2.4M is untouched by
it. 78 of the 240 floor rows are over-priced by more than $2M.

It is a fat tail, not a level shift — median floor-zone error is +$0.67M, the
90th percentile is +$5.83M, and the worst 12 rows carry 31% of the zone's
total error: Oladipo 2021 (+$24.9M), Oubre 2023 (+$18.1M), Harrell 2022
(+$17.2M), Drummond 2021 (+$16.4M), Whiteside 2020 (+$13.6M), Reggie Jackson
2020, Chris Paul 2025, Marc Gasol 2020, Blake Griffin 2021. Every one is the
same story: a player whose production still looks valuable taking a minimum
after injury or decline.

## The asymmetry you must respect

On the max side the false positives were caught by a **legal** rule — an
extension's raise cap meant the player could not reach the ceiling the push
aimed at. **There is no equivalent on the floor side.** No CBA rule raises a
player's floor by route; the minimum scale is the minimum scale. So a
pull-down has no structural brake, only the threshold and the selection rule.
Expect a tighter operating point than the max side's τ = 0.52, and do not
assume symmetry where the CBA does not provide it.

## Part 1 — diagnostics before any arm

1. **P(floor)** from the existing 6-class classifier (`route_mixture.py`),
   fold-honest, 10 seeds. Report AUC (phase-3 baseline: 0.8259), the
   calibration table, and the **purity curve** — the true-floor rate among
   rows with P(floor) ≥ τ, on the same grid the max sweep used.
2. **The honest headroom at each τ**, computed before any arm runs, exactly
   as the max sweep does it:
   - expected win = Σ over touched true-floor rows of the champion's current
     over-prediction;
   - expected collateral = Σ over touched non-floor rows of
     P × (champion prediction − floor).
   The pre-registered operating point is **the τ maximising (win − collateral)**.
3. **Report who the collateral is** at the selected τ — the named rows and
   their damage, the way the max sweep did. A pull-down false positive is a
   player predicted at the minimum who actually earned real money, and those
   errors are as large as the max side's.

## Part 2 — the arms

At the selected τ only (report the full sweep table, but gate only the
selected point):

- **A — pull only**: `pred = min(champ, floor_pct)` where P(floor) ≥ τ.
  The mirror of the max push.
- **B — pull with a margin**: `pred = champ + P·(0.95·floor_pct − champ)`
  where P ≥ τ, then the Stage-2 clip. The mirror of push-then-clip, with the
  margin FIXED at 0.95 and never tuned (the 1.05 precedent).
- **T — told**: pull exactly the rows that ARE at the floor. This is the
  Stage-3 told mode and it is scored in the same column per the 2026-07-26
  decision — but state the convention in the RESULT, and report the ex-ante
  arms beside it so the two are never confused.

Gates, on FIXED v7.13x rows: floor-zone MAE improves ≥ $0.30M (a lower bar
than the max side's $0.50M because the zone is 3.4× larger, so the same
per-row effect is worth more pooled); no non-floor segment's bias worsens by
more than $0.30M; C2 fixed segments ≤ $0.30M; B1 forward drop ≤ 0.003; paired
ΔSel t > 2 for the ex-ante arms.

## Part 3 — the question behind the question

The fat tail is fallen stars. Report whether the classifier can see them:
what P(floor) does the classifier give Oladipo 2021, Oubre 2023, Harrell
2022, Drummond 2021, Whiteside 2020? If it cannot identify them, say so with
the numbers — that is the same "the signal is not in the features" finding
the max side reached, and it tells the architect whether the remaining work
is a threshold or a data source (availability, injury history, age×decline
interactions are all in or near the feature set).

## Traps

- τ from the purity/objective rule only, never from a realized zone metric.
  Three experiments have died on that.
- The pull-down direction has no legal brake — over-pulling a player who
  earned real money is a $10M-class error, the same magnitude as the max
  side's collateral. The expected-collateral term in the selection rule is
  the only guard; do not weaken it.
- `floor_pct` is a recovered quantity (median at-floor pay per season ×
  experience bucket, ISSUES #6), not a published scale. Say what that
  imprecision costs on the touched rows.
- Do not touch the Stage-1 left-censoring (`floor_gate_k`, `sigma_left`) —
  this is an output-side experiment, and the censoring settings were swept and
  held at v7.8x.

## Deliverable

Branch + RESULT: the classifier diagnostics, the purity and objective curves
with the selected τ, the three arms at that τ with the full gate battery, the
named collateral, the fallen-star probability table, and a plain adopt /
do-not-adopt recommendation. Expected effort: a day.
