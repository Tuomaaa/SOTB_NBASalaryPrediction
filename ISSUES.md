# Open issues

Known problems that are real but were out of scope when found. Each entry is
written to be actionable without the conversation that produced it: what is
wrong, how to see it for yourself, what to do, and how to know you fixed it.

**If you find something you are not fixing right now, add it here** rather than
leaving it in a chat log. Delete an entry when it is fixed — this file is a
worklist, not a changelog. `VERSION_HISTORY.md` is where fixes get recorded.

Ordered roughly by how much damage each one does.

---

## 2. Continuation rows: the part deliberately left in

**Severity**: low — the provable and the corroborated cases are gone; what
remains is kept on purpose.

Closed over v7.6x-v7.7x. Six rows paid above their *tier* ceiling are provable
mislabels and go via `_filter_mislabeled_year1`. The broader class — ordinary
pre-2019 contracts whose first observed season wore a year-1 label — is demoted
by `_filter_continuations` on a three-signal consensus: the salary-matched
Spotrac span starts earlier, the season-over-season pay step is escalator-shaped
(0.92-1.081), and the row is absent from that season's FA-signings list. 119 rows
fell (2019: 35, tapering to 2026: 1); common-row A1 rose +0.0102. Spot-checked
top drops are verifiable mid-contract seasons (LeBron 2019, Simmons 2021,
Hayward 2023).

**A warning that must outlive this entry**: the span signal alone is NOT
sufficient. A span-only version deleted 346 rows, 7.2% of which sat in that
season's actual FA-signings list — Brunson 2022, VanVleet 2023, Jimmy Butler
2019, all genuine fresh signings. Spotrac's Free-Agent anchor is unreliable for
contracts later superseded by an extension. **Do not relax the consensus back to
span-only.**

What remains, deliberately: rows whose last-season pay is unobservable (2019
players outside the 155-player prehistory) and span-only suspects are KEPT.
Precision over recall — a stale price in training is cheaper than a deleted real
one. Expanding `salaries_prehistory.csv` (more cached BBRef player pages, parsed
offline by `scripts/backfill_prehistory_salaries.py`) converts more of them into
testable rows; a scraped signing date would settle them outright.

**The payoff is now quantified**: during the supply-feature ablation
(2026-07-23) a pure `is2019` dummy scored +0.0032 paired on the selection pool
(t = 2.38) — the 2019 slice still carries a season-level offset the features
cannot explain, exactly the signature of the stale prices left in. Cleaning
the remaining continuations is worth roughly that much; a season dummy itself
is not adoptable (it is memorization by construction — at inference every
future season scores 0 and it only launders training).

**Verify** (state at v7.8x): evaluation frame 1,172 rows; 2019 year-1 count 220
against 255 before; `check_caps.py` hit rates for 2019-2020 risen.

---

## 3. Single-seed and 10-seed CV R² are both called "the" CV R²

**Severity**: low — the site now leads with the forward number, not this one.

`src/model/train.py` writes `grabit_results.json` from a **single-seed** run;
`METHODOLOGY.md`, `PROJECT_BRIEF.md` and `evaluation_suite.json` quote the
**10-seed** average. Both are defensible numbers, but they are reported under
the same name. The site's headline is now the forward R² (out-of-sample on the
holdout season), which `export_web.py` cross-checks against
`evaluation_suite.json`; but `meta.json` still carries a secondary `cvR2` from a
single-seed fit on the training seasons, so the two-name ambiguity persists in
that field.

**Fix**: pick one as canonical — the 10-seed average is the one every document
uses — and have `train.py` write that into `grabit_results.json` so any consumer
inherits it. `export_web.py` no longer reads that file (it was rewritten to
forward-mode and dropped the `canon` guard for a rolling-origin cross-check), so
this is now only about keeping `train.py`'s own output honest.

---

## 4. Signing-mechanism labels: the residual 10%

**Severity**: low — the labels are diagnostics, never features.

Largely fixed 2026-07-23: `scripts/refresh_spotrac.py` rebuilt
`spotrac_signing_types.csv` with anchor-based season assignment (each contract
places itself at `fa_year − n … fa_year − 1` instead of the old backward walk,
which misaligned whenever a page skipped a deal), salary-aware disambiguation
for mid-season buyouts (Westbrook 2022 was a $46M supermax row labeled
"Minimum"), and player pages for extension signees harvested via the
upcoming-FA URL directory. Year-1 evaluation rows are now 85–91% labeled per
season (was ~43% overall); Minimum-labeled rows above $6M fell from 30 to 6.

What remains, for whoever next touches the labels:

- ~10% of evaluation rows are still Unknown — mostly two-way conversions and
  Exhibit-10s whose Spotrac pages carry no "Signed Using" block. Reproduce:
  `python scripts/refresh_spotrac.py --reparse-only` prints the per-season table.
- Six >$6M "Minimum" rows persist (Winslow 2019, Fultz 2019/20, Leonard 2020,
  Batum 2020, Noah 2020) — stretched/waived money colliding with a same-season
  minimum where the AAV-distance rule picks the wrong side.
- Extensions land under "Bird Rights" (n jumped 58 → 353), conflating
  re-signings with extensions. If the C2 mechanism slice is ever used to argue
  a retention premium, split these into their own category first.

Keep the field out of `FEATURE_COLS` — `METHODOLOGY.md` documents that
modelling it *lowers* CV R² by 0.0073; its value is as an out-of-sample
diagnostic.

---

## 5. Team and position missing for 5.4% of rows

**Severity**: low — cosmetic on the site, unused by the model.

167 of 3,113 rows have no `team_abbreviation` or `position` after the join
against `data/processed/training_data.csv` (still 5.4% as of the 2026-07-22
export). The valuation board renders these as "—". Neither field is a model
feature, so this affects presentation and the team filter only.

Root cause is the `team` column disagreeing between sources: the 2019-2024 rows
come from BBRef player salary pages and the 2025+ rows from team contract pages,
and a rebuild reproduces `team` on only 53.8% of rows (`position` on 94.6%) while
every one of the fourteen model features reproduces exactly. Cosmetic by
construction — but it means a team filter on the site silently omits rows rather
than showing them as unknown.

---

## 6. Two boundary rules rest on hand-curated or derived tables

**Severity**: low — both work today; both are maintenance the next refresh pays.

Stage 2's bounds are exact where the CBA is exact and approximate where the repo
lacks a source:

- **`early_supermax.csv`** enumerates designated-veteran deals signed two or more
  summers before they take effect (Wall 2019, Towns 2024), because no award
  window anchored to the start season can reach them. A blanket `s-3` lookback
  was tried at v7.4x and reverted at v7.5x — it un-censored two genuine 30% max
  signings. **Do not reintroduce a wider window.** The systematic fix is a
  signing date per contract, which Spotrac exposes and the scrape does not yet
  take; that would also let `_filter_continuations` drop its step-and-veto
  signals for a direct test.

  The signing dates are now scraped —
  `data/processed/contract_signing_dates.csv`, 3,169 dated signings — and they
  confirm both curated rows at a 2-summer gap (Wall signed 2017-07-26 for
  2019-2022, Towns 2022-07-07 for 2024-2027). Five further max-tier evaluation
  rows are also signed 2+ summers early (Booker 2024, Lillard 2021, Davis 2025,
  Harden 2019, Adebayo 2026), all extensions, and all already receive the
  correct tier ceiling through the award-window path — over-tier rows among
  them are 0, so nothing is broken today. `early_gap >= 2` in
  `parse_signing_dates.contract_spans()` is now a mechanical test that could
  replace both the curated list and the award window; that is a
  `_compute_max_eligible` change and wants its own dispatch.

  **The fourth payoff claimed here does not exist — struck 2026-07-23.** The
  exit-structure ablation found the expiring contract's length worth +0.0032
  paired (t=2.32) as a pure increment over the coverage artifact, and this
  entry predicted that transaction-level signing dates would lift coverage
  "toward ~95%" and unlock it. Measured, they do the opposite: the dated
  instrument sees *fewer* expiring contracts than the Spotrac blocks already
  do (a contract ending at S-1: blocks 60.7%, dates 50.8%, union 62.4%; any
  contract covering S-1: 87.3% / 79.4% / 88.1%). The blocks table knows 2,727
  distinct contracts against 1,953 dated ones, because a contract reaches the
  transactions list only if Spotrac logged the transaction while the contract
  history retains deals whose transaction entry it never had. Note also that
  the 54% quoted above could not be reproduced from any definition — the
  ablation script was a scratchpad experiment and is gone; the nearest
  construction is 60.7%. **The +0.003 stays locked, and signing dates are not
  the key to it.** Numbers in `docs/briefs/2026-07-23-signing-dates.RESULT.md`,
  target 3. RFA status, by contrast, tested empty once the artifact was removed
  (+0.0004), and option structure is unmeasurable historically — Spotrac
  annotates options on current contracts only (has_po is 0.000 for every
  fa_year before 2023 and 0.35-0.50 for 2028+).
- **`floor_pct`** is the median pay of at-floor rows per (season, experience
  bucket) — a recovery of the veteran-minimum scale from the data's own mass
  points, not the published scale. Buckets with few at-floor rows fall back to
  the season minimum. Accurate enough that the Stage-2 clip lands within
  $0.05-0.09M of observed pay, but a published scale would be exact.

**Verify**: `python -c "from src.model.train import *; ..."` — over-cap rows stay
0, and at-floor rows in the sub-2% predicted band keep bias under $0.10M.

---

## 8. 265 more continuation rows the three-signal filter does not reach

> **Addressed on branch `worker/continuation-filter-v2` (2026-07-24), pending
> architect landing** — spans fixed, residue remeasured to 223, filter swapped to
> a dated-span demotion (frame 1,172 → 949). Evidence:
> `docs/briefs/2026-07-24-continuation-filter-v2.RESULT.md`. Common-row A1 is
> neutral (−0.0009, t=−0.13) but the `is2019` offset is eliminated. One open
> item (JJJ 2025 renegotiation, see #14). Delete this entry when landed.

**Severity**: medium — 265 is an UPPER BOUND, not a count (see the architect
review below); the true residue sits in training wearing a year-1 label, and
it is now directly measurable rather than inferred.

With signing dates (`data/processed/contract_signing_dates.csv`, added
2026-07-23) a row's staleness is decidable outright: season S is a fresh price
when the covering contract's span STARTS at S, and a continuation when the span
starts earlier. Scored against `_filter_continuations` on the 1,291-row
pre-continuation frame:

| date verdict | filter kept | filter demoted |
|---|---|---|
| continuation | **265** | 111 |
| fresh (incl. early-signed first years) | 717 | 0 |
| undecidable (no covering contract) | 190 | 8 |

The filter's own precision is vindicated — **111 of its 119 demotions are
confirmed, 8 are undecidable, and 0 are contradicted.** The gap is entirely
recall. The 265 break down by contract year as 179 in year 2, 50 in year 3, 30
in year 4, 6 in year 5, and by season as 2019: 76, 2020: 33, 2021: 33, 2022: 22,
2023: 24, 2024: 26, 2025: 21, 2026: 30 — note 76 in 2019 alone against the
"~38+ suspected" ISSUES #2 records. The top cases by salary are Embiid 2023,
Butler 2023, Doncic 2022, Gobert 2021 — but see the architect review below:
those four are extension FIRST paying years misread as year 2, not
mid-contract seasons.

**Why the filter misses them**: most are extensions, absent from the FA-signings
list, whose season-over-season pay step falls outside the 0.92-1.081 escalator
band — an extension's first escalator year steps off a differently-based prior
salary. Two of the three signals fail, so the consensus never fires.

**Reproduce**: `python scripts/parse_signing_dates.py --cache-dir <cache>`, then
`contract_spans()` / `covering_contract()` from the same module against the
pre-continuation frame. Full table and the ten largest disagreements in
`docs/briefs/2026-07-23-signing-dates.RESULT.md`, target 2.

**Architect review (2026-07-23)** — the span derivation starts extensions one
year early, which inflates the 265 and misclassifies its head. Three
mechanisms, each verified against cap arithmetic on the top-10 table:

- **Option years pull the block anchor early.** Spotrac's `fa_year` for a
  deal whose final year is a player option is the option-decision summer, so
  `start = fa_year − years` begins the deal one season early. Embiid 2023 is
  $47.6M = 0.35 × the 2023 cap — the supermax's FIRST paying year, not
  "2 of 4"; same off-by-one for Gobert 2021 (vet-max year 1), Doncic 2022
  (30% Rose year 1), Butler 2023.
- **The signing-season fallback starts extensions a year early.** An
  extension begins paying the season AFTER it is signed: KD 2026 and
  Holmgren 2026 (both `match_confidence=none`, fallback spans) are first
  paying years read as "2 of 2" / "2 of 5".
- **Renegotiations re-price a season mid-span.** Markkanen 2024's $42.2M was
  set 2024-08-07 ("renegotiation-and-extend" in its own tx_text) — a fresh
  price wearing a year-4 span.

Under the standing convention — an extension's first paying year is a year-1
training row, and v7.7x kept every such row — **7 of the 10 largest
"disagreements" are correctly kept today.** The year-2 bucket (179 of 265) is
exactly where this false-positive class concentrates. The true residue is the
JJJ 2025 / Randle 2024 / Griffin 2019 class, extension years 2+; its size is
unknown until the spans are fixed.

**What to do, in order**: (1) fix `contract_spans()` — a final-year option in
the tx_text makes the span `[fa_year − years + 1, fa_year]`; an extension's
fallback span starts at `signing_season + 1`; a "renegotiat" match marks the
renegotiated season fresh. (2) Remeasure this table, with the head cases as
the acceptance test: Embiid 2023 / Doncic 2022 / Gobert 2021 / KD 2026 /
Holmgren 2026 read year 1, Markkanen 2024 reads fresh, JJJ 2025 / Randle 2024
/ Griffin 2019 stay continuations. (3) Only then the filter change: direct
span test where a date exists, three-signal consensus as the fallback. That
changes the training row count and so is a version-bump change, judged on
common-row A1 exactly as v7.7x was.

**Caveat before acting**: 19 of the 265 sit in that season's FA-signings list,
which is evidence against the date verdict on those rows — a ~7% error rate.
Do not demote a row whose FA-list membership contradicts the span without
resolving the conflict first; that disagreement is the same instrument clash
ISSUES #2 warns about, in the other direction.

**Verify**: after the span fix, the remeasured residue's head matches the
acceptance list above; after the filter change, confirmed demotions rise from
111 toward the remeasured count, the 2019 year-1 count falls from 220, and
the `is2019` control dummy (+0.0032, t=2.38) loses most of its remaining
signal, since that offset is the signature of exactly these stale prices.

---

## 9. 6.1% of Spotrac contract-block anchors precede their own signing date

**Severity**: low — now detectable and resolved in the one consumer that
exists, but any new consumer of block spans will hit it.

ISSUES #2 warns that "Spotrac's Free-Agent anchor is unreliable for contracts
later superseded by an extension". That is now counted rather than suspected:
of 1,953 dated contracts with a block-derived span, **120 (6.1%) have the block
anchor starting the contract BEFORE the transaction that signed it** — an
impossible span. The affected list is a roll-call of renegotiated stars: Kawhi
Leonard (signed 2021-08-12, anchor 2020), LeBron James (2018-07-09, anchor
2017), Jimmy Butler (2019-07-06, anchor 2018), Fred VanVleet (2023-07-07,
anchor 2022), Kyrie Irving (2025-07-06, anchor 2022).

These are the same rows that made a span-only continuation rule delete genuine
signings. `parse_signing_dates.contract_spans()` resolves them by trusting the
date — a contract cannot begin before it is signed — which restores VanVleet
2023, Butler 2019 and Brunson 2022 to year 1.

**Reproduce**: `contract_spans()` and read the `span_conflict` / `span_source`
columns; `date-resolved` marks the 120.

**What to do**: nothing urgent. But any code that reads `contract_start` from
`spotrac_signing_types.csv` inherits the bad anchor with no way to see it, so
prefer `contract_spans()` where a signing date exists. Note the limit of the
current resolution: it fires only when the anchor lands BEFORE the signing
date. The option-year off-by-one (#8, architect review) leaves the anchor at
or after the signing season and passes silently, so `contract_spans()` needs
the option-aware fix before any consumer trusts its year numbers.

**Verify**: `spans[spans.span_conflict].span_source.unique()` is
`['date-resolved']` only, and the three named signings read year 1.

---

## 10. Record the sigma sweep's mechanism and the relative C1 gate in METHODOLOGY

**Severity**: low — docs only; the decisions are made and the code already
matches them.

Old entry #7 closed 2026-07-23: a 36-config sweep plus boundary probe held the
incumbent censoring settings (0.02 / 0.55 / 2.0). Full evidence in
`docs/briefs/2026-07-23-sigma-gate-retune.RESULT.md` — do not re-run the sweep
on this row set. Two findings belong in METHODOLOGY (docs lane):

- **Sigma must never be selected on zone MAE.** Both censored sides are one-way
  valves: every zone row is biased toward its CBA bound and Stage 2's clip
  makes overshooting free, so zone MAE falls monotonically in sigma out to
  0.06 with no interior optimum, while the pinned-at-bound share climbs
  19% → 42% and the bill lands on the calibration slope (0.9883 → 0.9647 at
  sigma=0.04). Judged this way sigma degenerates into "how many rows to pin".
  gate_frac is inert on this row set (52-56 of 57 rows gated across
  0.45-0.65); the binding constraint is `is_max_contract`'s 0.90 threshold.
- **The C1 calibration gate is RELATIVE**, adjudicated 2026-07-23: a candidate
  is admissible when |slope − 1| exceeds the incumbent's by no more than
  0.005 (≈ $0.3M of scale distortion at a $60M max — C2's own materiality
  yardstick; paired slope-difference noise is far below it). The absolute
  [0.99, 1.01] window is dead — it excluded the champion's own 0.9883. Now
  encoded in `evaluate_suite.py`'s printed protocol block.

`_make_tobit_obj` also gained an inert `sigma_left` hook (bit-identical when
unset); the side-separation control it enabled showed the two censored sides
are independent and additive.

---

## 11. Spotrac 10-day contracts are parsed as 10-YEAR contracts

> **Fixed on branch `worker/continuation-filter-v2` (2026-07-24), pending
> architect landing.** `parse_contracts` now drops parsed length > 5 and never
> reads a year from a "day" field; reparse gives 5,769 rows, `contract_years.max()
> == 5`, `min(contract_start) == 2003`, filter still drops the same 119.
> **Correction to this entry**: the "2 at 6" are NOT 10-day money — they are the
> legitimate 6-year deals LeBron James 2010 ($109.8M) and Luol Deng 2008 ($71.0M)
> from the 2005 CBA. Both start pre-2019 and are read by no in-window consumer, so
> "drop > 5" removes them harmlessly. Delete this entry when landed.

**Severity**: low-moderate — the table is diagnostic-only, so no published
metric moves, but the bad rows feed `_filter_continuations`, which does.

`parse_contracts` in `scripts/scrape_spotrac_players.py` reads contract length
with `re.match(r"(\d+)\s*yr", val)` against the "Contract Terms:" field. Spotrac
writes a 10-day deal's terms in a form that leaves `10` as the captured number,
so the contract is recorded with `contract_years = 10`. The anchoring step then
expands it across ten fabricated seasons walking backwards from the anchor —
Anthony Tolliver has phantom contracts starting in 1980, 1990, 1998, 1999 and
2000; Alfonzo McKinnie in 1981 and 1991.

**Reproduce**:

```bash
python -c "import pandas as pd; c=pd.read_csv('data/processed/spotrac_signing_types.csv').drop_duplicates(['player_name_norm','contract_start','contract_years','total_value','aav']); print(c['contract_years'].value_counts().sort_index()); print(c[c.contract_years>5].head(20).to_string())"
```

328 of 2,727 distinct contracts have `contract_years > 5`, which is not a legal
NBA contract length in any era in the window — 326 at exactly 10, 2 at 6. Their
total values ($41k–$176k) and AAVs ($4k–$18k) are 10-day money, confirming the
reading. **20 of them carry a `contract_start` inside 2019–2026**, where they
can collide with real evaluation rows.

**Why it matters beyond the label**: `_filter_continuations` demotes a row when
a salary-matched covering contract starts earlier than the row's season. A
phantom 10-year span is exactly the shape that produces a spurious "starts
earlier" signal. The AAV-distance test (`aav_tol = 0.25`) screens most of them
out because 10-day AAVs are tiny, so the damage is probably zero today — but it
is zero by luck, not by construction.

**Fix**: reject the parse when the terms string does not actually say years.
Match `(\d+)\s*yr` only after confirming the field has no "day" token, and drop
any contract whose parsed length exceeds 5. Then re-run
`python scripts/refresh_spotrac.py --reparse-only` (no network). Cross-check:
`contract_signing_dates.csv` independently classes 344 ten-day transactions
via `contract_class` — the two instruments should agree on who the 10-day
players are.

**Verify**: `contract_years.max() <= 5` over the whole table; no contract has
`contract_start` before 1990; the continuation filter still drops 119 rows with
the same per-season split (2019: 35 … 2026: 1) — if that count moves, a phantom
span *was* load-bearing and the change needs a common-row A1 delta before it
lands. Natural home: fold into the #8 filter-v2 work, which touches the same
span machinery.

---

## 12. AAV is unavailable for half the evaluation frame, non-randomly

**Severity**: low — informational, and it bounds what any contract-structure
work can attempt.

Established by the 2026-07-23 AAV target-variable memo
(`docs/briefs/2026-07-23-aav-target-memo.RESULT.md`). Only **604 of 1,172**
evaluation rows (51.5%) can be assigned a contract whose total value reconciles
with the row's year-1 pay under a legal escalator. The missing half is not
random: coverage runs 79% above $25M and 27% below $3M, 81% on Bird Rights and
35% on Minimum-labelled rows. The covered subset's mean salary is $12.4M against
$5.5M for the uncovered.

This is the same collection-coverage channel the worker brief's gate 2 warns
about, and it means **any** feature derived from contract structure — length,
AAV, guarantee share, option structure — inherits a missingness pattern aligned
with player quality. Cost it as an increment over an explicit coverage-indicator
arm, never as a raw delta.

The root causes are the two above plus extensions that never appear on a
signed-FA page. (The memo predicted transaction-level signing dates would raise
coverage; the signing-dates shadow analysis, landed the same day, measured the
opposite — the dated instrument covers FEWER contracts than the blocks, union
gain under 2 points. See #8's source table and the RESULT's target 3.)

---

## 13. `prev_cap_pct` feature ignores `salaries_prehistory.csv`

**Severity**: low-moderate — `prev_cap_pct` is one of the 14 model features, so
a wrong value is a wrong feature, not merely cosmetic.

The FEATURE `prev_cap_pct` in `training_data_v2.csv` (built by
`scripts/phase3.build_contract_features`) is derived from that table alone, whose
rows start in 2019. When a row's prior season is 2018 or earlier the lookup
fails and `prev_cap_pct` gets a constant fill (**0.037227**) instead of the true
value — even though the salary IS on record. `_load_prev_season_cap_pct` in
`train.py` (used by `_filter_continuations`) already reads
`salaries_prehistory.csv` for exactly this boundary and gets it right, so the two
"prior pay" derivations disagree.

**Reproduce**: Klay Thompson 2019 carries `prev_cap_pct = 0.037227`; his 2018-19
salary is in `salaries_prehistory.csv` as `$18,988,725` = **0.186** of the 2018
cap, and `_load_prev_season_cap_pct()[("klay thompson", 2018)]` returns 0.186.
The same 0.037227 fill also appears on his 2020-2023 and 2025-2026 rows, whose
true priors ARE present, so the feature is broken for more than the 2018
boundary — `build_contract_features` should be audited, not just extended.

**Fix**: have `build_contract_features` fall back to `salaries_prehistory.csv`
for pre-2019 priors, the way `_load_prev_season_cap_pct` does, then re-check that
`prev_cap_pct` reproduces a player's actual prior-season cap_pct across a sample.
This is a feature-value change and wants its own dispatch (touches the frame all
14 features live in); measure a common-row A1 delta before landing.

**Verify**: Klay 2019 `prev_cap_pct` ≈ 0.186; no player carries the 0.037227 fill
where a real prior exists.

---

## 14. JJJ 2025 — is a renegotiated final year "fresh" or a "continuation"?

**Severity**: low — one training row; it does not move any headline (kept by both
the incumbent and the v2 filter). But it is an unresolved convention question the
architect owns.

The continuation-filter-v2 work (#8) treats a renegotiation-and-extend's signing
season as a fresh price (Markkanen 2024, Turner 2022) — the season is re-priced
to market even though an older span still covers it. **Jaren Jackson Jr. 2025**
is structurally identical: his 2021 extension is front-loaded/declining, and the
2025-07-13 Renegotiation-and-Extend raised his 2025-26 salary by $11,586,605 to
the observed $35.0M (yic=1 in the contract table). So the same rule reads it
**fresh**, but the 2026-07-24 brief's acceptance list expects it to *stay
continuation*. No mechanical rule separates JJJ 2025 from Markkanen 2024 /
Turner 2022. Decide whether a renegotiated final year is a fresh market price
(keep) or a mid-contract continuation (demote); the answer sets the convention
for all six renegotiation pairs. Evidence:
`docs/briefs/2026-07-24-continuation-filter-v2.RESULT.md` §4.1.
