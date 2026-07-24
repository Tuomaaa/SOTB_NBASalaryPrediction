# Task brief: real service years for the tier ceilings (ISSUES #20)

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.12x line, 944 rows, max zone n=68). Deliver a
branch + `docs/briefs/2026-07-26-service-years.RESULT.md`.

## Why

`_compute_max_eligible` derives service years as `season − draft_year`, and
falls back to **`age − 19`** where the draft table has no entry
([train.py:483](src/model/train.py:483)). That fallback covers **223 of 944
rows (24%)** and overstates service for anyone who entered late or undrafted,
pushing them into the 30% and 35% brackets.

An audit over the whole frame settles the scope. Of the **53 rows whose pay
lands exactly on a CBA tier** (within 1e-6 relative — the distribution is
knife-edge: widening the tolerance 10,000× adds only 4 rows), **exactly two
carry a ceiling above the tier they landed on**:

| row | pay | tier landed | ceiling granted | real service | fallback said |
|---|---|---|---|---|---|
| austin reaves 2026 | 25.000% | 25% | 30% | 5 (undrafted, debut 2021-22) | 8 |
| jimmy butler 2019 | 30.000% | 30% | 35% | 8 (drafted 2011) | 10 |

Both read `is_max_contract = False` today and both are real maxes.

## The rule this task must NOT implement

Do **not** set the ceiling from the observed salary ("pay lands on 25% →
ceiling = 25%"). The ceiling has to be computable **before the deal is
signed** — that property is what makes Stage 2 a legitimate inference-time
device (the AAV memo rejected a whole target change over it). A
salary-derived ceiling would (a) clip training predictions at the observed
pay, buying free accuracy from the target, and (b) be uncomputable in
`predict.py` for an unsigned free agent. The exact-tier audit is a
**detector**; the fix is to the inputs.

## Part 1 — a real first-NBA-season source

Add a per-player debut season, then define `service_years = season − debut`.
Preferred source: the cached BBRef player pages already used by
`scripts/height.py` (the player index / player page carries the career span),
parsed offline; live fetches only if the cache misses, at the standing 3s
rate limit, and a failed fetch degrades to missing-for-that-player, never to
a deleted row.

Do NOT use "first season observed in `salaries.csv`" as the primary source —
our salary window starts 2016 and would give Butler a debut of 2016
(service 3 in 2019 → 25% tier → a ceiling **below** his actual pay, which is
strictly worse than the current bug). It is acceptable only as a
cross-check for players whose debut is provably inside the window.

Keep `age − 19` as a last-resort fallback and **print how many rows still use
it**.

## Part 2 — rebuild and verify

Ceilings only; no other feature moves.

**Must hold:**
- Reaves 2026: `tier_ceiling_pct == 0.25`, `is_max_contract` True.
- Butler 2019: `tier_ceiling_pct == 0.30`, `is_max_contract` True.
- Max zone grows 68 → 70; frame stays **944** (a ceiling change must not
  move `_filter_mislabeled_year1`'s row count — if it does, report and stop).
- **Zero rows above their own ceiling** (`cap_pct > tier_ceiling_pct + 1e-4`)
  — the invariant the v7.4x ceiling audit established. A wrong debut year in
  the other direction breaks this, so it is the primary regression test.
- The 51 already-correct exact-tier rows keep their ceilings unchanged.
- Re-run the exact-tier audit after the fix: the "ceiling above the landed
  tier" count must be **0**.

**Report** (not gates — this is a correctness fix, adopt unless the paired
delta is significantly negative):
- How many rows' `tier_ceiling_pct` moved, in each direction, with the
  10 largest by salary and a one-line sanity note each.
- How many rows still fall back to `age − 19`.
- Paired ΔSel vs the same-run incumbent (10 seeds × 5 folds, per-fold — the
  **fold** is the pairing unit; a seed-paired t inflates ~3×, see the
  feature-batch RESULT §4), plus A1/A2/B1, C2 fixed segments, and the max
  zone MAE before/after.

## Part 3 — a note for the route-mixture line

Reaves 2026 is currently the second-largest collateral row in the phase-3
re-run (+$9.71M of push damage) purely because of this label. State in the
RESULT what the corrected labels do to the phase-3 collateral list (you can
read `p_max` from `scripts/eval_route_mixture_p2.py`'s OOF output if you
want the exact rows; recomputing the branch is NOT in scope).

## Traps

- Debut ≠ draft year: a drafted player who sat out a year debuts later, and
  service years count seasons on a roster. State which definition your source
  gives and spot-check five players by hand against a public profile.
- The `age − 19` fallback is load-bearing for 24% of rows today — a new
  source that covers fewer of them than the fallback "covers" is a
  regression; report coverage explicitly.
- Do not touch `designated_ineligible.csv` / `early_supermax.csv` (the #19 and
  supermax paths); this is the base-tier path only.
- Ceilings feed the censoring mask, so a moved ceiling changes what Stage 1
  treats as censored — that is intended, but it means the paired delta must
  be measured, not assumed.

## Deliverable

Branch + RESULT per the evidence-bundle format: the source and its coverage,
the moved-ceiling table, the invariant checks, the paired numbers, and the
phase-3 note. Expected effort: half a day.
