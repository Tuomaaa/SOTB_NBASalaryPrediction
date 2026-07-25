# Task brief: two features from the floor-crash review — role mismatch, and size-for-position

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-29-two-features.RESULT.md`.

Another worker is repairing the median-fill join and the buyout/invalid-row
filters from the same review. **Those rows are the ones your features would
otherwise be fitted against**, so if that work lands first the architect
re-runs yours at landing — pin where you are and do not rebase.

## Where these come from

`docs/briefs/2026-07-25-floor-crash-qualitative-review.md` reviewed all 34
floor-zone rows the champion over-prices by more than $5M and assigned each a
handling. Two of its categories are modelling hypotheses with named
mechanisms; everything else is a data repair, a filter, or a case the review
concluded should not be modelled at all. **You are building only these two.**

The review's own framing matters, because it is narrower than the feature
names suggest:

> **Low-quality high usage** (Oubre 2023 +$18.0M, Drummond 2021 +$16.3M,
> Mudiay 2019 +$8.1M): "high `USG%` combined with low relative TS%, low
> `AST%` and/or high `TOV%`". On Oubre: "20 points per game came with high
> usage, below-league scoring efficiency and very little playmaking. Teams
> did not view that weak-team role as portable to a contender. **The agreed
> modeling hypothesis is the joint signal**… not points per game alone."
> On Drummond: "a paint-bound center used 31.3% of possessions while
> finishing inefficiently and turning the ball over."

> **Position-relative height** (Harrell 2022 +$17.1M, Galloway 2020 +$5.4M):
> Harrell had "center-only offensive skills in a forward-sized body";
> Galloway "had point-guard height but shooting-guard function… every viable
> lineup needed another creator". The hypothesis is "position-season average
> height minus player height, kept general rather than a Harrell-specific
> rule".

## The prior rejection you must engage with, not ignore

METHODOLOGY's ablation table already records **`bpm`, `ts_plus`, `fg3_plus`,
`threepar_plus` at < +0.001, "Redundant with DARKO/LEBRON/RAPM"**. That is a
real result and it is about **main effects**. The review's hypothesis is a
**joint** one: usage is only damning *conditional on* low efficiency and low
playmaking, and a tree with 944 rows may not find a three-way interaction on
its own. Your job is to test the composite, and your RESULT must state
plainly whether it beats the already-rejected main effect — if `ts_plus`
alone and your composite score the same, the composite adds nothing and the
old rejection stands.

## Part 1 — role mismatch

Ingredients are all in `data/processed/impact_metrics.csv`: `usage_pct`,
`ts_plus`, `ast_pct`, `ast_tov_ratio`, `efg_plus`. Note `ts_plus` is **10.6%
null** — handle with native NaN, not a fill, and run the coverage-indicator
control the project requires when missingness might align with anything.

Build and test, in this order, so the increments are separable:

1. `ts_plus` alone — reproduce the old rejection on the current frame. If it
   no longer reproduces, that is itself a finding and the architect wants it
   before anything else.
2. The composite the review describes. Its exact form is yours to choose, but
   state it and justify it: something along the lines of usage above the
   median combined with efficiency and playmaking below, or a continuous
   product/score. **Do not tune the form against the metric** — pick a
   defensible construction, register it, then run it once. Trying five and
   keeping the best is the selection pressure three experiments have died on.
3. The composite plus its parts, so the RESULT can separate "the interaction
   matters" from "one part was doing the work".

## Part 2 — size for position

`position` exists in both `impact_metrics.csv` and `training_data_v2.csv`
(SG/PF/SF/PG/C, roughly balanced across 3,880 player-seasons) and
`height_inches` is already a feature. Build **position-season mean height
minus the player's height** — positive means undersized for the position.

Two cautions the project has earned:

- **Position labels here were derived from height at some point in this
  project's history** — check before you build. If `position` is a function
  of `height_inches`, then "position-mean height minus height" is a
  transformation of a feature the model already has, and its ablation is
  near-meaningless. Establish the provenance and say so; if the labels are
  height-derived, report that and stop Part 2 there.
- Use the **season** mean, not a global one — the league's positional heights
  drift.

## Judging

Full challenger gates, each arm independently:

- paired ΔSel ≥ +0.002 with **t > 2** on the selection pool, per-fold pairing
  (the fold is the unit; a seed-paired t inflates ~3×);
- A2 moves the same direction;
- C2 fixed-segment |bias| growth ≤ $0.30M;
- B1 forward drop ≤ 0.003;
- relative C1: |slope − 1| excess ≤ 0.005 vs the incumbent;
- coverage-indicator control for any arm whose missingness could align with
  era, team or player quality.

**SHIP-FORM testing is mandatory**: the column you score must be the column
`build_external_features` would produce, byte-for-byte. A native-NaN vs
median-fill difference has flipped a verdict in this project before.

## The segments that motivated this, reported for every arm

The whole point is the floor-zone tail, and the pooled metric will dilute it.
Report, on fixed rows:

- the five named rows (Oubre 2023, Drummond 2021, Mudiay 2019, Harrell 2022,
  Galloway 2020) — champion error and arm error, each;
- the floor zone (n=240, champion MAE $2.10M, bias +$2.04M);
- the 34 crash rows as a group;
- the standard C2 mechanism table.

**A feature that fixes the five and fails the pooled gate is a real result,
not a failure** — say so plainly and let the architect decide. The project
has a precedent for judging a targeted intervention where it acts, and it
also has a precedent for refusing to move a gate after seeing a score. Report
both numbers and argue for neither.

## Traps

- Do not touch the filter chain, the ceilings, `stages.py`, or the route
  machinery. Features only.
- Do not build a Harrell-specific or Oubre-specific rule. The review was
  explicit that the hypothesis is general; a rule that fires on five rows is
  a lookup table.
- Five of the 34 crash rows carry median-filled features (identical mpg
  21.96 / usage 17.2 / availability 0.67 — Harrell 2023, Cousins 2020, Bol
  Bol, Beasley, Beverley). **Exclude them from any interpretation of your
  segment numbers**; they are being repaired in parallel and their inputs are
  known-wrong.
- Deltas above +0.01 are a re-verify flag, not a victory lap.

## Deliverable

Branch + RESULT: the provenance check for `position`, the registered form of
the composite, the per-arm paired tables with per-fold values, the five named
rows before and after, the floor-zone and crash-group numbers, the ship-form
statement, and a plain adopt / do-not-adopt for each arm. Expected effort: a
day.
