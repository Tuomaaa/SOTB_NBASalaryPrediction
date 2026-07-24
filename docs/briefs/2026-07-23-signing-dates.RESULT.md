# RESULT — signing dates per contract

Worker evidence bundle for `docs/briefs/2026-07-23-signing-dates.md`.
Branch `worker/signing-dates`, cut from `master` at `6226c68`.
**Nothing is adopted here.** No existing table, filter, feature or model was
changed; the four analyses below are shadow reads.

---

## 1. What changed and why

`scripts/parse_signing_dates.py` parses the transactions list on all 627 cached
Spotrac player pages into a new table, `data/processed/contract_signing_dates.csv`
— 3,169 dated signings. The brief asked whether the contract blocks carry a
"Signed:" label that might cover more: they do not. A scan of every cached page
finds zero such labels; the blocks expose `Signed Using:`, `Contract Terms:`,
`Average Salary:` and `Free Agent:` only, so `ul.player-transactions` is the
sole date source and no choice between sources arises. The script never
fetches — it reads the cache the existing scrapers populate, so a player with
no cached page contributes no rows, which is missing-for-that-player and never
a deleted row elsewhere.

The one derivation everything downstream depends on is **the span**: which
seasons a dated contract covers. It is not the signing season plus length,
because an extension takes effect after it is signed. It comes from the matched
block anchor (`fa_year − years`), except where that anchor precedes the signing
date — a contract that began before it existed. 120 spans (6.1%) read that way,
and they are exactly the superseded shells ISSUES #2 warns the Free-Agent anchor
cannot be trusted on. **Resolving those to the signing date is the single most
load-bearing decision in this bundle**, and it is what makes VanVleet 2023,
Butler 2019 and Brunson 2022 — the three genuine signings ISSUES #2 records a
span-only rule wrongly cutting — read as year 1 again. `contract_spans()` and
`covering_contract()` are exported so the four consumers share one derivation.

---

## 2. Numbers

### 2a. Parse coverage

| | count |
|---|---|
| player pages parsed | 627 (all carry a transactions list) |
| transaction entries found | 8,444 |
| entries beginning "Signed" | 3,197 |
| parsed into a contract row | **3,196 (99.97%)** |
| of which via the loose fallback | 3 |
| unparseable text | **1** — `Signed with Spain` (an overseas move, correctly not an NBA contract) |
| unparseable dates | **0** |
| duplicate entries dropped | 27 (Spotrac lists some signings twice) |
| **rows written** | **3,169** |

Spotrac writes one date format, `Jul 06, 2025`; the abbreviated, dotted and
year-less forms the brief warned about do not occur. Every malformed entry was
a one-off typo in the source (`Signeda`, `$ 7.58 million`, `$$43.3 million`,
`contract wit New Orleans`, `Milwaukee( MIL)`) or a suffix form (`$838k`,
`$127.2M`). One class mattered: Spotrac drops the word "million" on ~1% of
entries (`$220.44 Designated Veteran Player extension`) and drops the word
"contract" on some extensions — including Towns's, one of this task's own
validation targets — so the match is anchored on the trailing `with <Team>
(ABB)` rather than on the word "contract".

### 2b. Match to `spotrac_signing_types.csv` blocks

Tolerance: contract length must be equal and total value within **2%** of the
block value. The text rounds (`$103.6 million` against `$103,608,840`), so the
test is relative; 2% is looser than the worst rounding (0.5%) and far tighter
than the gap between two contracts of the same player.

| confidence | n | |
|---|---|---|
| `exact` | 1,594 | value within 2% and years equal, uniquely |
| `value` | 1 | value matched, no year count in the text |
| `ambiguous` | 32 | several blocks agree; earliest `fa_year` taken, left flagged |
| `none` | 231 | money stated, no block agrees |
| `unmatchable` | 1,311 | no money in the text to key on |
| **matched of priced** | **1,627 / 1,858 = 87.6%** | |

The 1,311 `unmatchable` are two-way (452), 10-day (344), Exhibit-10 (227) and
rest-of-season (207) entries, which state neither years nor dollars. An earlier
cut matched those on years alone and returned 1,241 confident-looking
`ambiguous` hits built from nothing; requiring a stated value is what collapsed
ambiguity to 32.

`contract_class` guesses, so consumers can filter the non-market rows: standard
1,722, two-way 452, 10-day 344, exhibit-10 227, rest-of-season 207, rookie-scale
204, second-round 13.

Spans built: **1,953** — block-anchored 1,506, signing-season-assumed 327,
date-resolved (conflict) 120.

### Target 1 — early_supermax

Both curated rows confirmed, at exactly the 2-summer gap the entry describes:

| player | season | signed | span | gap | value |
|---|---|---|---|---|---|
| john wall | 2019 | 2017-07-26 | 2019–2022 | **2** | $171.0M / 4y |
| karl-anthony towns | 2024 | 2022-07-07 | 2024–2027 | **2** | $220.4M / 4y |

Seven max-tier evaluation rows are signed 2+ summers early; **five are not on
the curated list**:

| player | season | cap_pct | tier ceiling | signed | gap | curated |
|---|---|---|---|---|---|---|
| devin booker | 2024 | 0.35 | 0.35 | 2022-07-06 | 2 | no |
| karl-anthony towns | 2024 | 0.35 | 0.35 | 2022-07-07 | 2 | **yes** |
| damian lillard | 2021 | 0.35 | 0.35 | 2019-07-06 | 2 | no |
| anthony davis | 2025 | 0.35 | 0.35 | 2023-08-06 | 2 | no |
| james harden | 2019 | 0.35 | 0.35 | 2017-07-08 | 2 | no |
| john wall | 2019 | 0.35 | 0.35 | 2017-07-26 | 2 | **yes** |
| bam adebayo | 2026 | 0.30 | 0.30 | 2024-07-06 | 2 | no |

All seven are extensions. **No ceiling is presently wrong**: over-tier rows
among them = 0, because the award-window path in `_compute_max_eligible`
already reaches the five uncurated ones. So the curated list is incomplete as
an *enumeration of early-signed max deals* but is not currently *missing* a
row it needs — a distinction that matters, because `early_gap >= 2` is now a
mechanical test that could replace both the curated list and the award-window
guess. That is a Stage-2 change and therefore the architect's call.

### Target 2 — continuation-filter shadow test

Verdict on the 1,291 pre-continuation rows against `_filter_continuations`:

| date verdict | filter kept | filter demoted |
|---|---|---|
| continuation | **265** | 111 |
| fresh | 572 | 0 |
| fresh-early | 145 | 0 |
| unknown (no covering contract) | 190 | 8 |

- decided rows: 1,093 of 1,291 (84.7%); **agreement 828 / 1,093 = 75.8%**
- **filter demotions confirmed: 111 of 119, 8 undecidable, 0 contradicted.**
  The three-signal consensus has not demoted a single row the dates call fresh
  — its precision is exactly what v7.7x claimed.
- the disagreement is entirely one-directional: **265 rows the dates call
  stale that the filter kept.** By contract year within the covering deal:
  year 2 → 179, year 3 → 50, year 4 → 30, year 5 → 6. Total pay $1,959M,
  median $2.2M — a long tail of small contracts with a heavy head.
- by season: 2019 → 76, 2020 → 33, 2021 → 33, 2022 → 22, 2023 → 24, 2024 → 26,
  2025 → 21, 2026 → 30.

Ten largest-salary disagreements, all *dates say continuation, filter kept*:

| player | season | salary | signed | span | year | ext |
|---|---|---|---|---|---|---|
| joel embiid | 2023 | $47.6M | 2021-08-17 | 2022–2025 | 2 of 4 | yes |
| jimmy butler | 2023 | $45.2M | 2021-08-07 | 2022–2024 | 2 of 3 | yes |
| kevin durant | 2026 | $43.9M | 2025-10-19 | 2025–2026 | 2 of 2 | yes |
| lauri markkanen | 2024 | $42.2M | 2021-08-28 | 2021–2024 | 4 of 4 | no |
| chet holmgren | 2026 | $41.5M | 2025-07-13 | 2025–2029 | 2 of 5 | yes |
| luka doncic | 2022 | $37.1M | 2021-08-10 | 2021–2025 | 2 of 5 | yes |
| rudy gobert | 2021 | $35.3M | 2020-12-20 | 2020–2024 | 2 of 5 | yes |
| jaren jackson jr. | 2025 | $35.0M | 2021-10-18 | 2022–2025 | 4 of 4 | yes |
| blake griffin | 2019 | $34.2M | 2017-07-17 | 2017–2021 | 3 of 5 | no |
| julius randle | 2024 | $33.1M | 2021-08-26 | 2022–2025 | 3 of 4 | yes |

**My reading of who is right: the dates, on all ten.** Each is a verifiable
mid-contract season of a deal signed years earlier — Embiid's 2021 supermax
extension running into its second year in 2023, Doncic's 2021 rookie-max into
its second in 2022. None is a fresh price, and each is sitting in training
wearing a year-1 label. The filter kept them because it demands three
independent signals and at least one fails on each: most are extensions absent
from the FA-signings list whose season-over-season pay step falls outside the
0.92–1.081 escalator band, since an extension's first escalator year can step
by more than 8% off a differently-based prior salary.

Confidence caveat: 19 of the 265 sit in that season's FA-signings list, which
is evidence *against* my verdict on those rows — call it a ~7% error rate on
this set. **I did not change the filter**; row-count changes are an escalation
trigger and this is the architect's to land.

### Target 3 — expiring-contract length coverage

**This target's premise does not survive measurement.** ISSUES #6 and the brief
expect dated transactions to push coverage "toward ~95%". They do not — the
dated instrument covers *fewer* expiring contracts than the blocks already do,
and the union adds under 2 points.

| definition | blocks | dates | union |
|---|---|---|---|
| any contract covering season S−1 | 87.3% | 79.4% | **88.1%** |
| a contract *ending* at S−1 (i.e. actually expiring) | 60.7% | 50.8% | **62.4%** |

By season, union of "ending at S−1": 2019 49.5%, 2020 58.1%, 2021 64.8%,
2022 63.8%, 2023 65.7%, 2024 67.4%, 2025 66.4%, 2026 72.4%.

I could **not reproduce the 54% baseline** the brief quotes. The exit-structure
ablation that produced it was an experiment script and, per the worker rule
that experiments live in a scratchpad, it is not in the repo, so its exact
denominator is unrecoverable. The nearest definition I can construct is
"blocks, contract ending at S−1" = 60.7%. Whichever definition the architect
intended, the comparison that matters is measured on one definition at a time,
and on every one of them **dates lose to blocks**.

Mechanism, in one sentence: the blocks table knows 2,727 distinct contracts
against 1,953 dated ones with a year count, because a contract appears in the
transactions list only if Spotrac logged the transaction, while the contract
history section retains deals whose transaction entry it never had — the 136
rows the blocks see and the dates do not are concentrated at previous-season
2018 (35 of 136), the oldest edge of the evaluation window.

**Consequence: the +0.0032 exit-structure signal stays locked.** Signing dates
are not the key to it, and ISSUES #6's fourth payoff should be struck.

### Target 4 — extension vs re-sign inside Bird Rights

The split ISSUES #4 asked for:

| | n |
|---|---|
| Bird-Rights evaluation rows | 309 |
| of which dated | 304 (98.4%) |
| **extensions** | **162 (53.3%)** |
| re-signings | 142 |

By season (extension share): 2019 30.0% (n=40), 2020 59.1% (22), 2021 50.0%
(32), 2022 58.8% (34), 2023 51.0% (49), 2024 60.9% (46), 2025 64.3% (42),
2026 53.8% (39).

The two halves are not the same population: mean `cap_pct` is **0.180 for
extensions against 0.130 for re-signings**. ISSUES #4's warning is confirmed —
if the C2 mechanism slice is ever used to argue a retention premium, this
category must be split first, because "Bird Rights" is currently a 53/47 blend
of two groups paid 5 percentage points of the cap apart.

Extension share elsewhere, as a control that the label is doing real work:
Early Bird 6.2%, Cap Space 1.8%, MLE 1.2%, Minimum 0.8%, Sign & Trade 0.0%,
Non-Bird 0.0%. The signal is concentrated in Bird Rights exactly as expected.

---

## 3. Gate verdicts

**Not applicable** — this is a data task. Nothing enters `FEATURE_COLS`, no
filter changed, no row count moved, and no CV number was computed or claimed.
The evaluation frame reproduces at 1,172 rows at this branch's HEAD, unchanged.

---

## 4. Anomalies

1. **The 265-row disagreement in target 2 is the headline and it is large** —
   more than double the 119 rows the filter removes. It is not "too good": it
   is one-directional, every top case is independently verifiable, and it
   points at *under*-removal, which is the conservative failure the filter was
   deliberately tuned for ("precision over recall"). But ISSUES #2 estimates
   "~38+ suspected 2019 continuations" and I find 76 in 2019 alone, so the
   hidden class is roughly twice the size that entry records.
2. **Span conflicts are a measurement of ISSUES #2's hazard, not noise.**
   6.1% of block anchors precede their own signing date. The affected list is
   a roll-call of renegotiated stars (Kawhi 2021, LeBron 2018/2010, Butler
   2019, VanVleet 2023, Kyrie 2025). This is the first direct count of that
   class the repo has.
3. **Target 3 contradicts the brief's own premise.** I flag this rather than
   soften it: an expected payoff written into ISSUES #6 does not exist.
4. **The 54% baseline is unreproducible** (section above). No claim in this
   bundle rests on it.
5. **Not verified**: whether the seven early-signed max deals would still get
   correct ceilings if `early_supermax.csv` were replaced by an `early_gap >= 2`
   rule. That touches `_compute_max_eligible`, which the worker brief puts
   behind an escalation, so I measured the inputs and stopped.
6. Machine note, not a finding: `import pandas` under the sandboxed Bash tool
   took 90s–4min per invocation on this box while PowerShell served the same
   import in 0.94s. Everything here was run through PowerShell with
   `OMP_NUM_THREADS=6`. Several early runs were killed mid-flight by that
   contention; all reported numbers come from completed runs.

---

## 5. Files touched

Branch **`worker/signing-dates`** at
`C:\Users\panh3\Documents\ROSE\Personal Project\BBall-worker-signing-dates`
(git worktree, cut from `master` @ `6226c68`).

| file | status |
|---|---|
| `scripts/parse_signing_dates.py` | new — parser + `contract_spans()` / `covering_contract()` |
| `data/processed/contract_signing_dates.csv` | new — 3,169 rows |
| `docs/briefs/2026-07-23-signing-dates.RESULT.md` | new — this file |
| `ISSUES.md` | amended — see §7 |

Reproduce (raw HTML is gitignored, so a fresh worktree must point at the
cache in the main checkout):

```bash
python scripts/parse_signing_dates.py --cache-dir "../BBall Analysis/data/raw/html_cache/spotrac_players"
```

Shadow analyses live in the session scratchpad, not the repo, per the worker
rule on experiment scripts: `validate3.py` produces every number in §2, plus
`target2_verdicts.csv`, `target3_coverage.csv`, `target4_labels.csv`.

---

## 6. Proposed commit message

> Signing dates per contract from the cached Spotrac transactions
>
> Parse ul.player-transactions on all 627 cached player pages into
> data/processed/contract_signing_dates.csv — 3,169 dated signings, one
> unparseable entry ("Signed with Spain"), zero unparseable dates. The
> contract blocks carry no date field at all, so the transactions list is the
> only source; the parser never fetches.
>
> A span is derived from the matched block anchor, except where that anchor
> precedes the signing date — 6.1% of spans, the superseded shells ISSUES #2
> warns about. Those resolve to the signing date, which restores VanVleet
> 2023, Butler 2019 and Brunson 2022 to year 1.
>
> Four shadow reads, nothing adopted: both early_supermax rows confirmed at a
> 2-summer gap (five further early-signed max deals exist, all already
> correctly ceilinged); _filter_continuations demotions are 111/119 confirmed
> and 0 contradicted, but 265 further rows read as mid-contract; the expiring-
> contract coverage the dates add is negative against the blocks, so the
> exit-structure signal stays locked; Bird Rights splits 162 extensions to
> 142 re-signings, paid 0.180 vs 0.130 cap_pct.

Version number deliberately unclaimed — no published metric moves, and the
assignment is the architect's.

---

## 7. ISSUES.md additions

Three edits, all appended or narrowly corrective; no other agent's lane
touched.

- **new #8** — the 265-row continuation residue, with the reproduce command
  and the ~7% FA-list caveat.
- **new #9** — span conflicts as a measured 6.1% of block anchors, and the
  date-resolution rule that fixes them.
- **#6, fourth payoff** — struck, with a pointer to target 3's numbers. Left
  the rest of the entry intact; the curated-list and `floor_pct` maintenance
  points still stand.
