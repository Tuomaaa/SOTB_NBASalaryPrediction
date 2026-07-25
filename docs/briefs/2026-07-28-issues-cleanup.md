# Task brief: clean up ISSUES.md — duplicate numbers, fixed entries, stale titles

Read `docs/worker-brief.md` first. This is a **bookkeeping** task with one
research component. You own exactly one file, `ISSUES.md`, and you change no
code, no data and no other document.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-28-issues-cleanup.RESULT.md` (short).

**Two other workers are out** (stage-3 wiring, ceiling consistency). Both are
forbidden from touching `ISSUES.md` and will instead put their proposed
entries in their own RESULTs. **You will therefore not have their text.** Do
not try to guess it; the architect merges it at landing. Say in your RESULT
what the final numbering is so the architect can slot theirs in without
another collision.

## The state

`grep -nE "^## [0-9]+\." ISSUES.md` currently gives:

```
2, 3, 4, 5, 6, 12, 17, 18, 20, 20, 21, 25, 26, 27, 24, 25
```

Two `#20`, two `#25`, and the order is not monotonic. On top of that,
several entries describe problems that are already fixed — the file's own
header says "Delete an entry when it is fixed — this file is a worklist, not
a changelog."

## Part 1 — decide, per entry, and show your work

For every entry, establish from the repo (git log, the RESULT bundles under
`docs/briefs/`, the tag messages, and the code itself) whether it is **fixed**,
**partly fixed**, or **open**. The file is the next agent's first read, so a
wrong call here costs more than it looks.

Known cases, stated so you can verify rather than discover:

- **#3** (single-seed vs 10-seed CV under one name) — the cleanup-debt worker
  landed a fix on 2026-07-26. Verify `train.py` writes the 10-seed figure,
  then delete.
- **#18** (awards name join drops footnote-marked stars) — fixed in the same
  landing (`norm()` strips footnote marks; 148 rows gained award mass).
  Verify, then delete.
- **#20 (the `age − 19` service fallback)** — fixed by the service-years work
  (v7.13x, debut seasons scraped, 99.5% coverage). Verify Reaves 2026 and
  Butler 2019 read the right tiers, then delete.
- **#21** (veteran-extension raise caps "are not implemented") — the caps
  **are** implemented (`extension_cap.py`, landed 33325e9). The title is now
  the opposite of the truth. Either delete it or rewrite it to describe what
  remains; say which and why.
- **#20 (the three zone-gate protocol defects)** and **#25 (the τ-objective's
  collateral term)** — both are protocol findings from the floor-branch work,
  both still open, and one of them needs a new number.
- **#26**, **#27** — open at the time of writing, but the ceiling-consistency
  worker is fixing both right now. **Leave them in place** and renumber them
  like any other open entry; the architect deletes them at that worker's
  landing. Note this explicitly in your RESULT so nobody thinks you missed it.
- **#24** (`cba_era` and the extension multiple use different CBA boundaries,
  both correct) — open documentation debt, keep.

## Part 2 — renumber, and make it hard to break again

Renumber the survivors into a single monotonic sequence starting at 1, and
**update every cross-reference in the repo that names an issue number**. Those
exist in `docs/briefs/*.RESULT.md`, `docs/QUEUE.md`, `CLAUDE.md`,
`METHODOLOGY.md` and in code comments — `grep -rn "ISSUES #"` finds them. This
is the part most likely to be done badly: a renumber that leaves dangling
references is worse than the duplicates.

**If the cross-reference count is large**, do not renumber. Instead keep the
existing numbers, resolve only the two collisions by giving the newer entry of
each pair the next free number, and say in your RESULT that a full renumber
was declined and why. A stable wrong-looking number beats a broken reference.
State the count you found either way — that number is the decision.

Then add a one-line convention at the top of the file: numbers are permanent
and never reused, new entries take `max + 1`. That is what would have
prevented both collisions.

## Part 3 — the ordering claim

The header says entries are "ordered roughly by how much damage each one
does". Check whether that is still true and reorder if it is not, or delete
the claim if severity is no longer the organising principle. Do not leave a
statement in the file that the file itself contradicts.

## Traps

- **Verify before deleting.** An entry that reads fixed but is not is how a
  bug comes back. For each deletion, run the entry's own **Verify** block and
  paste the output in your RESULT.
- Do not delete an entry merely because a fix was *dispatched*; only because
  it is *landed and verified* on `master`.
- Do not touch any file other than `ISSUES.md` and your own RESULT — **except**
  cross-reference updates in Part 2, which are the one sanctioned exception,
  and only if you renumber.
- No code, no data, no rebuilds.

## Deliverable

Branch + a short RESULT: the per-entry verdict table with the verification
output for each deletion, the renumber decision with the cross-reference count
that drove it, the final numbering, and a note on where the two in-flight
workers' entries should slot in. Expected effort: half a day.
