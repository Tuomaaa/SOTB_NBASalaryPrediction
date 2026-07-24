# Task brief: continuation filter v2 — fix the spans, then swap the filter

Covers ISSUES #8 (the residue) and #11 (the 10-day parse bug). Read
`docs/worker-brief.md` first — especially the isolation section: work in your
own `git worktree`, cap threads at `OMP_NUM_THREADS=6`.

**Pin**: branch off current `master` in your own worktree. Deliver a branch +
`docs/briefs/2026-07-24-continuation-filter-v2.RESULT.md`.

## Why

v7.7x's three-signal consensus removes 119 stale rows with proven precision —
the signing dates confirmed 111 of 119 and contradicted 0. Its recall has a
real gap (mid-contract seasons of extensions survive, because an extension is
absent from FA lists and its first escalator step breaks the 0.92–1.081 band),
and the dates can close it. But the dates' span derivation currently starts
extensions one year early, so the raw "265 missed" count is inflated with
false positives. ISSUES #8's architect-review block is the spec for this task:
fix the spans first, remeasure, and only then touch the filter.

## Part 1 — span and parse fixes

In `scripts/parse_signing_dates.py::contract_spans()`:

1. **Option-aware anchor.** When a matched contract's final year is an option
   (tx_text or the block mentions a Player/Club Option on the last season),
   Spotrac's `fa_year` is the option-decision summer, so the span is
   `[fa_year − years + 1, fa_year]` — today's `[fa−years, fa−1]` starts these
   one season early. State your option-detection rule and how many spans move.
2. **Extension fallback +1.** An unmatched extension
   (`is_extension=1, match_confidence=none`) starts paying at
   `signing_season + 1`, not `signing_season`.
3. **Renegotiation carve-out.** When tx_text contains `renegotiat`, the
   renegotiated season is a FRESH price for that season regardless of the
   covering span (Markkanen 2024, Turner 2022).

In `scripts/scrape_spotrac_players.py` (ISSUES #11): parse `(\d+)\s*yr` only
when the terms string has no "day" token; drop any parsed length > 5; rerun
`python scripts/refresh_spotrac.py --reparse-only` (offline). Run #11's verify
block, and cross-check the affected players against
`contract_signing_dates.csv`'s `contract_class` (344 ten-day transactions —
the two instruments should agree).

## Part 2 — remeasure, with an acceptance list as the go/no-go

Re-run the signing-dates RESULT's target-2 comparison on the fixed spans.
All of the following must hold before Part 3:

**Read year 1 / fresh:** Embiid 2023, Doncic 2022, Gobert 2021, Butler 2023,
Durant 2026, Holmgren 2026, VanVleet 2023, Butler 2019, Brunson 2022,
Markkanen 2024 (renegotiation), **Jaren Jackson Jr. 2026** (early extension —
user-confirmed), **Trae Young 2026** (user-confirmed genuine 2026 signing; if
the instruments read otherwise, escalate with exactly what the cached page
shows — do not demote).

**Stay continuation:** Jaren Jackson Jr. 2025, Randle 2024, Griffin 2019,
Hayward 2019, Love 2022, Millsap 2019.

Report the remeasured residue table (by contract year within the covering
deal, and by season). Expect materially fewer than 265; the year-2 bucket
(179) is where the span fixes bite.

## Part 3 — the filter swap

In `_filter_continuations`, for rows where a covering dated contract exists:
demote when the covering span starts before season S, **unless** (a) the
season is renegotiated-fresh, or (b) the row appears in that season's
FA-signings list — an FA-list appearance contradicting the span is the
ISSUES #2 instrument clash; keep the row and list every such conflict in the
RESULT. Rows without date coverage keep the three-signal consensus unchanged.

## Judging — the row count changes, so v7.7x's protocol applies

- **Common-row paired A1** is the headline: score the incumbent frame's model
  and the new frame's model on the COMMON rows, same folds and seeds, per-fold
  deltas. (v7.7x's adoption cleared +0.0102 on this instrument.)
- Guardrails on common rows: C2 fixed-segment |bias| growth ≤ $0.3M; B1
  forward reported for both frames (context, not a gate — the row sets
  differ).
- Zone row lists change with the frame. Report the new zone sizes and MAEs,
  and explicitly do NOT compare them to the old $6.11M / $1.98M — different
  rows (D1).
- ISSUES #8's verify: the 2019 year-1 count falls from 220; measure the
  `is2019` dummy's paired delta on the new frame — its +0.0032 (t=2.38)
  should shrink if the removed rows were the season-offset carrier.
- **Nothing is adopted on your branch.** If the architect adopts, that landing
  is v7.9x and the tag is the architect's.

## Side-report (measure, do not fix)

`klay thompson 2019` carries `prev_cap_pct = 0.037` in the frame — he earned
roughly 17% of the cap in 2018-19, so this looks like a median fill over a
missing prehistory salary. Check whether his 2018 salary exists upstream and
report what you find (one paragraph; ISSUES #6's lane owns the fix).

## Traps

- Do NOT relax the FA-list veto to chase a bigger demotion count — the
  Brunson/VanVleet/Butler false positives are the standing warning.
- An extension's first paying year is a year-1 row by standing convention. A
  "cleaner" definition that demotes them is a modeling-policy change and is
  not yours to make; the acceptance list encodes the convention.
- R² across different row sets is never comparable — every headline number is
  common-row paired or it does not go in the RESULT.
- The pre-continuation frame is 1,291 rows. If Part 1 changes that count,
  something upstream broke — stop and escalate.
- Deltas above +0.01 on common-row A1 are a red flag to re-verify, not a
  victory lap (worker-brief rule).

## Deliverable

Branch + RESULT per the evidence-bundle format: span-fix counts, the
acceptance table with per-row readings, the remeasured residue table, the
FA-list conflict list, common-row paired numbers with per-fold deltas, the C2
table, the `is2019` delta, and the Klay paragraph. Expected effort: a day.
