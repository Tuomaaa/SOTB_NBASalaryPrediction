# Task brief: one membership definition — rookie contracts out, and the board reuses the chain

Read `docs/worker-brief.md` first — own worktree, `OMP_NUM_THREADS=6`.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-29-membership-one-definition.RESULT.md`.

**This task changes the training row count.** `worker-brief.md` tells you to
escalate when that happens; it is escalated already and this brief is the
decision. What it does *not* authorise is changing the target, the features, or
the evaluation protocol — if you find yourself needing any of those, stop.

Two other workers are out (docs pass 2; the waiver interaction). The waiver
worker is pinned to the 944-row frame; the architect re-runs its arm on your
frame at landing. **Do not rebase and do not coordinate with them.**

## The decision, and where it came from

The user spotted second-round rookies (Maxime Raynaud, Jaylen Wells) on the
Signing Board and asked why. The investigation found two halves of one problem,
and the user's ruling covers both: *a first contract cannot be predicted, so
throw it out along with the mid-season ones.*

**Half one — the board admits rows training rejects.** `export_web.py:757`
decides membership on two conditions:

```python
signing = (out["year_in_contract"] == 1) & ~out["is_rookie_scale"]
```

Training decides it with a five-filter chain. They disagree on **612 of 1,556**
board rows. `export_web.py:1035` then scores the site's advertised forward
accuracy on that same loose rule, so for 2026:

| row set | n | R² | MAE | bias |
|---|---:|---:|---:|---:|
| board's rule (what the site quotes) | 124 | **0.8146** | $2.96M | −$0.64M |
| the same rows, filter chain applied | 95 | **0.7987** | $3.42M | −$1.24M |
| the 29 the chain rejects, alone | 29 | −0.5584 | $1.49M | +$1.35M |

The 29 raise the public number by **+0.016** while scoring −0.56 among
themselves — they sit near the minimum, far below the mean, so they contribute
denominator and almost no error. A second-round contract is not rookie-scale, so
`~is_rookie_scale` cannot see it. Across all seasons the display damage is worse
than the second-round case: the surplus ranking is topped by **prorated
mid-season signings**, whose partial-season pay is compared against a
full-season prediction — Dinwiddie 2023 paid $1.55M and shown at $23.79M for a
$22.24M "surplus"; Aldridge 2020, Will Barton 2022, Beverley 2022, Roberson
2020 behind him.

**Half two — training still keeps 70 rookie contracts.** First-rounders are
removed by the rookie-scale filter, but second-round and undrafted first
contracts survive it, because neither signs a rookie-scale deal. Those 70 rows
carry **4.2% of the frame's variance and 1.0% of its squared error**: the model
gets them nearly exactly right (MAE $0.53M / $0.99M against the frame's $2.74M)
because they are all the same number by convention, not because it priced
anything. Removing them takes A1 **0.8210 → 0.8144**.

## Part 1 — the predicate

The architect measured six candidates. Two results decide it:

- **The Spotrac `second-round-exception` label is not usable.** 83 rows in
  `spotrac_signing_types.csv`, **1** of them in the 944-row frame.
- **Service years work and take nothing they should not.** `exp <= 1` removes 70
  rows whose best-paid member is $3.81M (Daniss Jenkins), with everything else
  in the $2–3M minimum band. `exp == 0` removes only 4.

**Registered predicate: the player's first two NBA seasons** — `exp <= 1`, where
`exp = season - debut_season`.

Three things you must get right, and the third is the interesting one:

1. **Use `train.py`'s own name cleaning.** `_compute_max_eligible` maps debut
   seasons through `pn_clean`, not `player_name_norm`. The architect's scratch
   measurement skipped that and produced 76 spurious NaNs (Towns, JJJ, SGA).
   Your row count will differ from 70 once the join is right — report the true
   number, do not reproduce 70 for its own sake.
2. **Decide the unknown-debut rows explicitly.** State the default and the
   count; keeping them is the safe direction, but say so rather than letting a
   `NaN <= 1` silently do it.
3. **Explain why these contracts land at `exp == 1` rather than 0.** Only 4 rows
   sit at `exp == 0`, which is not what "first contract" should look like. Either
   the salary rows begin a season after the debut, or these are genuinely second
   seasons and the predicate is catching year 2 of a two-year rookie deal. Those
   are different facts with different filters. **Establish which before you
   filter anything** — if it is the second, the predicate may need to be the
   *first contract* rather than the *first two seasons*, and that is a change to
   this brief that you should escalate rather than make.

Name it in the chain's style (`_filter_rookie_contracts`), place it where its
print statement joins the others, and have it report its drop count and season
breakdown like the neighbours do.

## Part 2 — the bridge, and it is the deliverable that matters most

**R² is not comparable across row sets.** Every published number from v1.0 to
v8.2x was computed on a frame that includes these rows. After this lands, A1
falls to roughly 0.814 — **that is a denominator change, not a regression**, and
a reader who sees 0.8210 → 0.814 in `VERSION_HISTORY.md` without the bridge will
read it as the model getting worse.

Produce:

- the **current champion scored on both frames**, unchanged in every other
  respect: A1 / A2 / B1 on the 944-row frame and on the new one, side by side;
- the **common-row delta** — champion vs champion on the rows both frames share,
  which must be ~0.0000, since nothing about the model changed. If it is not
  zero, something else moved and you must find it before continuing;
- the same pair for the **baseline ladder**, so the lift-over-baseline numbers
  stay interpretable;
- a short paragraph the docs agent can lift verbatim, stating the two frames and
  which published numbers belong to which.

Do NOT relabel or rewrite `VERSION_HISTORY.md` — that is the docs lane. Give
them the paragraph and the table.

## Part 3 — make the board reuse the chain

`export_web.py` must stop re-deriving membership. Both the display mask
(`:757`) and `fwd_metrics` (`:1035`) take the filter chain's output.

Then decide, explicitly and in the module docstring beside the existing `is_fa`
note, which board each rejected class belongs on:

- a second-round pick on the minimum genuinely **is** a surplus asset, so the
  Value Board may keep him — he just cannot enter a *pricing accuracy* number;
- a **prorated mid-season signing** should not appear in the surplus ranking at
  all, or must be annualised before it does. A $22.24M surplus that is really a
  four-month contract is not a finding.

Report the board's new 2026 `n` and forward R², and confirm they reproduce the
suite's 2026 origin on the same rows.

## Part 4 — regenerate the export (ISSUES #33)

`outputs/web/valuations_export.csv` is dated 2026-07-25 and reproduces v8.0x;
v8.1x and v8.2x have landed since and moved the 2026 origin to 0.8499. Re-run
`scripts/export_web.py` **after** Parts 1–3, so the regeneration does not bake
the old membership rule in again.

## Traps

- Do not touch the target, the feature list, `stages.py`, the ceilings, or the
  route machinery.
- Do not touch `ISSUES.md` numbering — numbers are permanent, next is 35.
  Entries #32, #33 and #34 are this work; delete #32 and #33 only if you have
  actually satisfied their "Fixed when" clauses, and leave #34 alone.
- **A1 will fall. That is the expected result.** Do not treat it as a failure and
  do not go looking for something to offset it. The number that has to be clean
  is the common-row delta in Part 2.
- `scripts/check_caps.py` after anything that touches loading.
- The 2019 season is the data's first, so "first season present in the data" is
  not a rookie test — it sweeps in Vince Carter, Haslem and Jimmy Butler. The
  architect made exactly this mistake; use debut seasons, not data extent.

## Deliverable

Branch + RESULT: the `exp == 1` provenance answer, the predicate's true drop
count and its removed-row list, the both-frames bridge table, the common-row
zero check, the board's new numbers, the regenerated export's headline, and the
docs paragraph. Expected effort: a day.
