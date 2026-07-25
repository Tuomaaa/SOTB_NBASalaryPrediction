# Task brief: does an offseason-injury signal exist and can we get it?

Read `docs/worker-brief.md` first — isolation, own worktree,
`OMP_NUM_THREADS=6`.

**This is a RECONNAISSANCE task**, in the mould of the MLE mass-point and AAV
memos: you establish whether the information exists and can be obtained, and
you build no feature. "It does not exist" is a complete and valuable answer.
Three other workers are out (stage-3 wiring, ceiling consistency, ISSUES
cleanup); you touch none of their files. You may add cached HTML and a new
curated CSV under `data/raw/`, and your own RESULT. **Do not touch
`ISSUES.md`** — put your entries in the RESULT.

**Pin**: current `master`. Deliver a branch +
`docs/briefs/2026-07-28-offseason-injury-probe.RESULT.md`.

## Why

The floor branch (`2026-07-27-floor-branch.RESULT.md`) is the largest measured
headroom in the project — pulling true at-floor rows onto their floor is worth
**+0.0396 of A1** — and it is unreachable because the classifier cannot see
which players will take a minimum. The reason is precise and worth restating,
because it defines what would count as a useful signal:

> The twelve worst floor rows score **above the average NON-floor row** on
> minutes, all three impact metrics, usage, prior pay and awards.
> `availability_3yr` reads **0.616 for them against 0.540 for the rest of the
> zone** — higher, not lower — because they were healthy and productive in the
> season being priced and got hurt in the summer *after*.

So the missing fact is dated **between the end of the priced season and the
signing**. Oladipo tore a quad tendon in May 2021 and signed for $2.4M that
August; nothing in his 2020-21 box score says so.

**The two injury files we already have cannot supply it**, and the architect
checked before writing this brief:

| file | span | records in Jun-Sep |
|---|---|---|
| `injuries_2010-2020.csv` | 2010-10-03 … 2020-10-06 | **1.88%** |
| `Injury Database - Oct 2021 - June 2024 (1).csv` | 2021-10-19 … 2024-06-17 | **0.18%** |

They are game-day availability logs — "player X is Out for game Y" — so with
no games there are no rows. On top of that there are two holes against our
2019-2026 window: **Oct 2020 → Oct 2021 is entirely absent** (which covers
Oladipo 2021, the single largest floor error), and nothing exists after June
2024 (so 2025 and 2026 are uncovered). The players themselves are present
(Oladipo 174 rows, Chris Paul 65), so name matching is not the obstacle —
the offseason is.

## Part 1 — the existence check, on nine named rows

Before any scraping design, establish by hand whether the fact is even
recorded anywhere public, for the nine rows that carry the headroom:

| row | pay | champion over-prediction |
|---|---|---|
| victor oladipo 2021 | $2.39M | +$24.92M |
| kelly oubre jr. 2023 | $2.02M | +$18.13M |
| montrezl harrell 2022 | $2.46M | +$17.20M |
| andre drummond 2021 | $2.40M | +$16.36M |
| hassan whiteside 2020 | $1.62M | +$13.58M |
| reggie jackson 2020 | $1.62M | +$9.95M |
| chris paul 2025 | $2.30M | +$9.67M |
| marc gasol 2020 | $2.56M | +$9.37M |
| blake griffin 2021 | $2.64M | +$8.31M |

For each: was there a **dated** event between the end of the priced season and
the signing date (`contract_signing_dates.csv` has the signing date) that a
data source records — surgery, a season-ending injury in the prior playoffs, a
publicly reported rehab timeline? Report **date, event, and where you found
it**, one line each.

**The finding that matters most is the negative one.** If some of these nine
have no injury at all — Harrell may simply have hit a cold market, Gasol left
for Spain — then injury data cannot explain them, and the ceiling on this
whole line drops accordingly. **Count them.** A signal that explains four of
nine is worth a quarter of what one explaining all nine is worth, and the
architect needs that fraction before funding a scrape.

## Part 2 — sources, and what each would cost

For whichever fraction Part 1 shows is recorded, evaluate the routes:

1. **The Spotrac player pages we already cache** (627 of them, used for
   signing dates) — do their transaction lists carry injury or surgery
   entries, or only contract events? This is the cheapest possible answer
   because the pages are already on disk; check it first.
2. **Pro Sports Transactions** or an equivalent public transactions archive —
   these historically log offseason surgeries and "placed on IL" events.
   Report coverage across 2019-2026, whether the offseason months are
   populated, and the scraping courtesy constraints.
3. Anything else you find. State licensing/robots constraints honestly.

For the leading candidate, report: seasons covered, share of our 944 rows it
could annotate, and whether it would fill the 2020-21 and post-2024 holes that
sink the files we have.

## Part 3 — the ceiling, computed before anyone builds anything

Assume a perfect offseason-injury flag on the fraction Part 1 established.
Compute the oracle: what does A1 become if the floor branch could pull exactly
those rows onto their floor?

The full-zone oracle is +0.0396 over 240 rows. Your number is the subset
version — the same computation restricted to rows a perfect injury flag would
identify. **Compute it from `outputs/models/oof_reference.csv` directly**; you
do not need to fit anything. If it comes out small, say so plainly: that is
the answer, and it saves the project a scrape.

## Traps

- Do not build a feature and do not touch `FEATURE_COLS`. This brief ends at
  "here is what exists, here is what it would be worth".
- Scraping courtesy per CLAUDE.md: cache everything, 3s+ between live
  requests, a failed fetch degrades to missing-for-that-player and never to a
  deleted row.
- The nine rows above are the *known* fat tail, but the zone has 78 rows
  over-priced by more than $2M. If Part 1 goes well, say what a sample of the
  next tier looks like — the nine may not be representative.
- Beware the mirror trap the floor RESULT names: an injury flag that is only
  available *after* the fact is a told-parameter, and by the 2026-07-27
  convention a told parameter earns its place only when the salary still needs
  a non-trivial computation afterwards. An offseason injury dated **before**
  the signing is genuinely ex ante and passes; one dated after does not.
  State which kind each of your nine events is.

## Deliverable

Branch + RESULT: the nine-row existence table with dates and sources, the
count that did NOT have a recorded event, the source comparison with coverage
figures, the subset oracle, and a plain go / no-go on funding the scrape.
Expected effort: half a day.
