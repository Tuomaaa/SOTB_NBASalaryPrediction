# RESULT: offseason-injury probe

**Branch**: `injury-probe` off `master` (5fb89b1)
**Date**: 2026-07-25
**Task**: reconnaissance — does a dated offseason-injury signal exist for the
floor zone's fat tail, and what would it be worth?

---

## Part 1 — existence check on the nine named rows

For each row the brief names, I looked for a **dated** event between the end of
the priced season and the signing date (from `contract_signing_dates.csv`) that
a public source records — surgery, season-ending injury in the prior playoffs,
or a publicly reported rehab timeline.

| # | player | season | signing date | dated offseason event? | date | event | source | ex ante? |
|---|--------|--------|-------------|----------------------|------|-------|--------|----------|
| 1 | victor oladipo | 2021 | 2021-08-07 | **YES** | 2021-04-08 (injury) / 2021-05-12 (surgery) | Ruptured right quad tendon vs LAL; season-ending surgery in New York | ESPN, NBA.com | yes — injury and surgery both predate the signing |
| 2 | kelly oubre jr. | 2023 | 2023-09-26 | **NO** | — | Hand surgery was Jan 2023 (in-season, returned). No offseason event. Unsigned until late September; reporting attributes it to a cold market and wing surplus on Charlotte, not injury | NBA.com, CBS Sports | n/a |
| 3 | montrezl harrell | 2022 | 2022-09-13 | **NO (off-court)** | 2022-05-12 | Arrested in Kentucky; felony marijuana trafficking charge (3 lbs). Reduced to misdemeanor Aug 2022. Dated and ex ante, but NOT an injury | NBC Sports, CBS Sports | yes (dated, ex ante) but not injury |
| 4 | andre drummond | 2021 | 2021-08-04 | **NO** | — | Buyout from Cleveland, rest-of-season with Lakers. No surgery, no reported injury. Market decline after two minimum-deal seasons | NBA.com, ESPN | n/a |
| 5 | hassan whiteside | 2020 | 2020-11-27 | **NO** | — | No injury. Posted 16.3/14.2/3.1 in Portland. Market limited by playstyle (non-spacing big) and effort reputation | NBC Sports Bay Area, Kings Herald | n/a |
| 6 | reggie jackson | 2020 | 2020-12-01 | **NO** | — | Buyout from Detroit Feb 2020, rest-of-season with Clippers, re-signed for minimum. No injury reported | CBS Sports, Hoops Rumors | n/a |
| 7 | chris paul | 2025 | 2025-07-21 | **NO** | — | Minor finger sprain during 2024-25 season (not offseason). Age 40, dramatic decline. Signed as a reserve backup | Yahoo Sports, NBA.com | n/a |
| 8 | marc gasol | 2020 | 2020-11-24 | **NO** | — | Hamstring injury was Dec 2019 (in-season, returned). Age 35, worst statistical season. Market was simply small | ESPN, Silver Screen and Roll | n/a |
| 9 | blake griffin | 2021 | 2021-08-09 | **NO (chronic)** | — | Arthroscopic left knee surgery Apr 2019 and debridement Jan 2020 — both over a year before this signing. Chronic knee decline, not a discrete offseason event. Buyout from Detroit Mar 2021, rest-of-season with Brooklyn, re-signed Aug 2021 | NBA.com, CBS Sports | n/a — no event in the window |

### The count

**1 of 9** rows has a dated offseason injury: Victor Oladipo.

One additional row (Harrell) has a dated off-court event (drug arrest) that is
ex ante and plausibly depressed his market, but is not an injury and would
require a broader "offseason adverse event" framing rather than an injury flag.

The remaining 7 rows have **no recorded offseason event at all**. Their
minimum-salary outcomes are explained by: age/decline (Gasol, Chris Paul,
Griffin), playstyle devaluation (Whiteside), buyout market dynamics (Drummond,
Jackson), and a cold free-agent market (Oubre). These are structural factors
the model's existing features attempt to price — the model overpredicts because
the features say these players are worth more, and the features are not wrong
about their on-court value; the players simply accepted less for reasons the
feature set cannot see.

**The critical implication**: an offseason-injury flag, even a perfect one,
cannot explain the fat tail. It explains the single largest row but not the
pattern.

### The "next tier" — a sample beyond the nine

The brief asks whether the nine are representative. The floor zone has **63
selection-pool rows** overpredicted by more than $2M and **30** by more than $5M.
A sample of the next tier outside the nine:

| player | season | over-prediction | offseason injury? |
|--------|--------|----------------|-------------------|
| javonte green | 2024 | +$9.58M | Possibly — came back from a prior-season injury, went unsigned, entered G League |
| tyus jones | 2024 | +$9.23M | No — cold market, wanted starter role, settled for minimum |
| gary trent jr. | 2024 | +$9.18M | No — turned down $15M/yr offer, market dried up |
| kendrick nunn | 2020 | +$8.24M | No — undrafted, market-value ceiling |
| demarcus cousins | 2020 | +$5.82M | **Yes** — ACL tear Aug 2019 (working out in Vegas), predate signing |

Only Cousins in this next-five sample has a clear offseason injury. Trent and
Jones are market-dynamics / player-choice cases. The pattern holds: most
at-floor overpredictions are explained by non-injury factors.

---

## Part 2 — sources and what each would cost

### Source 1: Spotrac player pages (already cached, 627 pages)

**Checked first as the brief instructs.** The cached pages' transaction lists
carry **only contract events** — signings, extensions, trades, waivers. The
parser (`scripts/parse_signing_dates.py`) filters for lines starting with
"Signed" and discards the rest. A scan of the extraction regexes confirms no
injury, surgery, or "placed on IL" patterns are captured.

The raw transaction lists *may* contain non-signing entries (Spotrac does list
some injury-related transactions on player pages), but the existing cache was
populated to extract contract data, and even if injury lines exist on some
pages, the coverage would be limited to the 627 players already cached — a
subset of the full floor zone.

**Verdict**: cheapest to check but unlikely to carry the needed signal. Would
require a re-parse of the 627 cached pages looking for non-"Signed" transaction
entries, and even then coverage is uncertain.

### Source 2: Pro Sports Transactions (prosportstransactions.com)

This is the canonical public transactions archive for NBA. It logs "placed on
injured list", surgery announcements, and movement transactions year-round,
including the offseason. Key properties:

- **Coverage**: comprehensive across 2010-2026, including offseason months
- **Offseason entries**: unlike our two existing injury CSVs (which are
  game-day availability logs with <2% of entries in Jun-Sep), PST logs
  transactions whenever they occur — a July surgery announcement gets an entry
- **Scraping**: the site has a search interface at
  `prosportstransactions.com/basketball/Search/Search.php`, but is behind
  Cloudflare bot protection, so automated scraping would need to work around
  that (or use the PyPI package `pro_sports_transactions`)
- **Courtesy**: no explicit robots.txt restrictions found, but the Cloudflare
  wall suggests they prefer human access

**Verdict**: the best candidate for coverage. Would fill the 2020-21 hole and
the post-2024 gap that sink our existing files. The PyPI package
`pro_sports_transactions` exists and may avoid the Cloudflare issue. However,
see Part 3 — the ceiling is too low to justify the scrape.

### Source 3: Basketball Reference injury page

BBRef has an injuries page (`basketball-reference.com/friv/injuries.fcgi`) but
it is a current-day snapshot, not a historical archive. It cannot provide
offseason injury data for past seasons.

### Source 4: our existing injury CSVs

As the brief establishes: `injuries_2010-2020.csv` and the Oct 2021 – June 2024
file are game-day availability logs. Only 1.88% and 0.18% of their entries
fall in Jun-Sep. They cannot supply offseason injuries by construction, and
they have coverage holes at 2020-21 and post-2024.

---

## Part 3 — the ceiling, computed before anyone builds anything

All numbers computed from `outputs/models/oof_reference.csv` directly, using
the evaluation frame's `is_at_floor` flag and `floor_pct`. The oracle replaces
a row's champion OOF prediction with its `floor_pct` (= the minimum the player
would actually be paid), simulating a perfect classifier that identifies that
row as at-floor.

Drummond 2021 and Whiteside 2020 are in the **confirmation split** and excluded
from the selection pool. The selection-pool analysis uses the remaining 7 of
the 9 named rows.

| subset | rows | oracle A1 | delta vs champion | % of full floor headroom |
|--------|------|-----------|-------------------|--------------------------|
| Champion (baseline) | 803 | 0.7928 | — | — |
| **Oladipo only** (1 confirmed injury) | 1 | 0.8020 | **+0.0092** | 24.5% |
| Oladipo + Cousins (from next tier) | 2 | 0.8026 | +0.0097 | 25.9% |
| All 7 named sel-pool rows | 7 | 0.8135 | +0.0207 | 55.2% |
| Full floor zone | 203 | 0.8303 | +0.0375 | 100% |

A perfect offseason-injury flag that identifies **only** the confirmed-injury
rows (Oladipo, and Cousins from the next tier) is worth **+0.0097 of A1**, or
about **a quarter** of the full floor-zone headroom.

The other five named selection-pool rows (Oubre, Harrell, Jackson, Chris Paul,
Gasol, Griffin) collectively carry +0.0110 additional headroom — but they have
**no offseason injury**, so an injury flag cannot reach them.

---

## Go / no-go on funding the scrape

**No-go.**

The reasoning:

1. **The signal is too sparse.** Of the 9 rows carrying the most headroom, only
   1 has a dated offseason injury. Including the next tier (Cousins), 2 of the
   top 14 floor-zone overpredictions are injury-explained. The rest are age
   decline, market dynamics, legal issues, and player choice.

2. **The ceiling is small.** A perfect injury flag on the 2 confirmed cases is
   worth +0.0097 of A1. At gate 1's bar of +0.002 with t > 2, this is above
   threshold on the oracle, but the oracle assumes perfect identification. A
   realistic classifier would identify a subset of offseason injuries (not all
   of which lead to minimum contracts) and produce a much smaller realized gain.

3. **The ceiling-to-cost ratio is poor.** Pro Sports Transactions is behind
   Cloudflare, the PyPI package's maintenance status is unknown, and the scrape
   would need to cover ~944 player-seasons across 8 years of offseason months.
   A half-day scraping and cleaning effort for a ceiling of +0.0097 (best
   case), likely realized at +0.002–0.004, is not competitive with other open
   headroom.

4. **The floor zone's real problem is not injury.** 7 of the 9 fat-tail rows
   are explained by factors the model's features can see (age, decline,
   performance) but the model still overprices. This suggests the floor
   classifier's bottleneck is the *decision* to take a minimum — often a player
   choice or market-timing outcome — not a missing injury fact. An injury flag
   addresses at most a quarter of the headroom.

---

## Findings for ISSUES.md (not written there per brief instruction)

No new issues discovered. The floor zone's composition is consistent with the
floor branch RESULT's characterization: the twelve worst rows "score above the
average non-floor row on minutes, all three impact metrics, usage, prior pay
and awards." This probe confirms that for 7 of 9, the missing information is
not a datable injury but a structural reason to accept less — which is harder
to encode and may represent an irreducible component of the floor zone's error.

---

## Files touched

- `docs/briefs/2026-07-28-offseason-injury-probe.RESULT.md` — this file (new)

No code, data, or model files were modified.
