# Task brief: signing dates per contract

Read `docs/worker-brief.md` first — especially the isolation section. This is
a DATA task: you produce new artifacts and shadow analyses; you change no
existing table, no filter, no model. Four downstream consumers are waiting on
this data, and each gives you a built-in validation target.

**Pin**: branch off current `master` in your own worktree. Deliver a branch.

## Goal

A signing date for every Spotrac contract we know about, as a NEW file
`data/processed/contract_signing_dates.csv` with columns:

    player_slug, signing_date, contract_years, total_value, is_extension,
    fa_year_matched (nullable), match_confidence

`fa_year_matched` links each dated transaction to the contract blocks already
parsed in `data/processed/spotrac_signing_types.csv` (match on years + total
value proximity; state your tolerance and report the match rate).

## Where the dates live

The 627 cached player pages (`data/raw/html_cache/spotrac_players/`) carry a
transactions section (`ul.player-transactions`) with entries like
"Signed a 4 year $104 million contract with New York (NYK) - July 12, 2022"
and "... contract extension ...". Parse date, years, amount, and the
extension keyword from the text. Contract blocks may also carry a "Signed:"
label — check one page by hand and use whichever source covers more.
**Cache first**: you should need little or no live scraping; if you must
fetch, 5s delays and backoff per `config.py`, and a failed fetch degrades to
missing-for-that-player, never to a deleted row.

## The four validation targets (all shadow analyses — report, change nothing)

1. **early_supermax check**: the two rows in
   `data/raw/raw_external/early_supermax.csv` (Wall, Towns) should show
   signing dates 2+ summers before their start seasons. Do they? Any OTHER
   max-tier contract in the data with a 2+ summer gap that the curated list
   misses?
2. **Continuation-filter shadow test**: with dates, "fresh signing for season
   S" means a covering contract signed between the previous season's end and
   season S's start. Compare row-by-row against the current three-signal
   `_filter_continuations` verdicts on the evaluation frame: agree/disagree
   counts, and the 10 largest-salary disagreements with your reading of who
   is right. DO NOT change the filter — row-count changes are an escalation
   trigger; the architect lands that separately.
3. **prev_contract_years coverage**: with transaction-derived contract
   records, what share of the 1,172 evaluation rows can see their expiring
   contract's length? (Current Spotrac-block coverage: 54%; the locked
   signal needs ~90%+.) Report overall and by season.
4. **Extension vs re-sign**: share of Bird-Rights-labeled evaluation rows
   whose contract text says "extension" — the split ISSUES #4 wants.

## Known traps

- Superseded shells: a renegotiated deal leaves its old block on the page
  with an unreliable fa anchor (ISSUES #2's standing warning) — transaction
  text is your friend here, blocks are not
- 10-day contracts, two-ways, Exhibit-10s produce transaction entries too;
  keep them but flag with a `contract_class` guess so consumers can filter
- Date parsing: month names, "Jan." abbreviations, missing years — report
  your unparseable-entry count rather than silently dropping

## Deliverable

Branch + `docs/briefs/2026-07-23-signing-dates.RESULT.md` per the evidence-
bundle format (numbers table = the four validation targets' results; gate
verdicts not applicable — nothing is adopted here). Include the parse
coverage stats: pages parsed, transactions found, dated, matched to blocks.
Expected effort: half a day to a day.
