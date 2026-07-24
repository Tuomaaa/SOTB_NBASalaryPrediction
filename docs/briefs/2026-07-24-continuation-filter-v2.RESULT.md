# RESULT — continuation filter v2 (ISSUES #8 + #11)

Worker evidence bundle for `docs/briefs/2026-07-24-continuation-filter-v2.md`.
Branch `worker/continuation-filter-v2`, cut from `master` at `cf682ed`.
**Nothing is adopted here.** The architect reviews this evidence and rules; if
adopted, that landing is v7.9x and the tag is the architect's.

One item on the Part-2 acceptance list resolves against the brief's stated
expectation (**JJJ 2025**) and is escalated in §Anomalies, not resolved
unilaterally. It does not change the headline (the row is kept in both frames).

---

## 1. What changed and why

Two data-layer bugs were fixed, then the continuation filter was swapped from a
three-signal consensus to a dated-span test. **(a) The 10-day parse bug (ISSUES
#11):** `parse_contracts` read a 10-day deal's terms string `10 yr(s) / $85,578`
as a 10-YEAR contract, fabricating phantom spans back to 1980. The fix rejects
any parsed length above the 5-year CBA maximum (and never reads a year count
from a field that says "day"). **(b) The span derivation (ISSUES #8):**
`contract_spans()` started option-final extensions one season early (Spotrac's
`fa_year` for an option-final deal is the option-DECISION summer), started
unmatched extensions in their signing season rather than the season after, and
had no notion that a renegotiation re-prices the current season. All three are
fixed; the option correction keys on the option season carried in the
transaction text and self-restricts to exactly the deals Spotrac anchored on the
option.

With correct spans, `_filter_continuations` now demotes a row **outright when a
dated contract covers it and starts before the season**, unless the season is
renegotiated-fresh or the row appears in that season's FA-signings list; rows the
dates cannot decide fall back to the unchanged three-signal consensus. This
closes the recall gap ISSUES #8 measured (extensions' mid-contract years, whose
escalator step falls outside the 0.92-1.08 band the old filter required). The
training frame drops from **1,172 → 949 rows**. On the 949 common rows the
change is **CV-neutral** (selection paired ΔA1 = −0.0009, t = −0.13; MAE
unchanged at $3.21M), while the `is2019` season-offset signal that ISSUES #8
flagged as the signature of the remaining stale prices **is eliminated**
(+0.0035, t=2.29 on the old frame → −0.0004, t=−0.91 on the new). The common-row
instrument is by construction blind to the benefit of *removing* stale rows (it
scores only rows both frames keep); the `is2019` collapse is where that benefit
shows up. This is therefore a correctness change at no CV cost, not a CV win.

---

## 2. Numbers

### 2a. Part 1 — span & parse fix counts

**10-day parse (ISSUES #11).** `spotrac_signing_types.csv` 8,954 → 5,769 rows.
328 phantom `contract_years > 5` contracts removed (326 at 10yr, 2 at 6yr);
`contract_years.max()` now 5, `min(contract_start)` now 2003 (was 1980), 0
contracts start before 1990. Cross-check against the dated instrument: 146
players carried a phantom >5yr contract, **144 also carry a `contract_class ==
"10-day"` transaction** in `contract_signing_dates.csv` (which classes 344
ten-day transactions across 148 players) — the two instruments agree. The two
non-overlaps are the **2 six-year contracts, which are legitimate** (LeBron 2010
Miami $109.8M/6y, Luol Deng 2008 Bulls $71.0M/6y — 2005-CBA-era deals, not
10-day money as ISSUES #11 states); both start pre-2019 and are read by no
in-window consumer, so the spec'd "drop > 5" is harmless (see §4).
**The continuation filter still drops the same 119 rows with the identical
per-season split (2019:35 … 2026:1) after the reparse** → no phantom span was
load-bearing (ISSUES #11 verify).

**Span fixes (`contract_spans`), on 1,953 dated contracts.**

| fix | detection | spans affected |
|---|---|---|
| option-aware anchor | `OPTION_YEAR_RE` = `(\d{4})(?:-\d{2})?\s+(?:Player\|Club\|Team)\s+Option`; slide the span to end on the option season when `option_year > naive_span_end` | fires on **49**; **14 change span_start** (Embiid/Doncic/Gobert/Butler + 10 more); the other 35 already had that start via the date-conflict resolver, and the rule corrects their `span_source` |
| extension fallback +1 | `is_extension=1` and unmatched → start = `signing_season + 1` | **24** (Durant 2026, Holmgren 2026, …) |
| renegotiation flag | `tx_text` contains `renegotiat` → `(player, signing_season)` fresh | **6** pairs (Turner 2022, Sabonis 2023, Clarkson 2023, Isaac 2024, Markkanen 2024, JJJ 2025) |

Net: **38 spans move start** (30 touching 2019-2026). `span_source` counts:
block 1,494, assumed 329, date-resolved **81** (was 120 — option-shift now
resolves the VanVleet/Brunson class that previously read date-resolved),
option-shifted 49. **ISSUES #9 invariant preserved**: every `span_conflict` row
is `date-resolved` only, and the three named signings read year 1 (VanVleet 2023
[2023-2025], Butler 2019 [2019-2022], Brunson 2022 [2022-2025]).

### 2b. Part 2 — acceptance table (per-row readings on the fixed spans)

Verdict key: `fresh` = covering span starts at S; `reneg-fresh`/`fa-fresh` =
covering span starts earlier but overridden fresh; `continuation` = span starts
earlier; `undecidable` = no dated contract covers the row.

**EXPECT fresh (read year 1):**

| player | S | verdict | covering span | salary | pass |
|---|---|---|---|---|---|
| joel embiid | 2023 | fresh | 2023-2026 | $47.6M | ✓ |
| luka doncic | 2022 | fresh | 2022-2026 | $37.1M | ✓ |
| rudy gobert | 2021 | fresh | 2021-2025 | $35.3M | ✓ |
| jimmy butler | 2023 | fresh | 2023-2025 | $45.2M | ✓ |
| kevin durant | 2026 | fresh | 2026-2027 | $43.9M | ✓ |
| chet holmgren | 2026 | fresh | 2026-2030 | $41.5M | ✓ |
| fred vanvleet | 2023 | fresh | 2023-2025 | $40.8M | ✓ |
| jimmy butler | 2019 | fresh | 2019-2022 | $32.7M | ✓ |
| jalen brunson | 2022 | fresh | 2022-2025 | $27.7M | ✓ |
| lauri markkanen | 2024 | reneg-fresh | 2021-2024 | $42.2M | ✓ |
| jaren jackson jr. | 2026 | fresh | 2026-2029 | $49.0M | ✓ |
| trae young | 2026 | fresh | 2026-2029 | $49.5M | ✓ |

**EXPECT continuation (stay stale):**

| player | S | verdict | covering span | salary | pass |
|---|---|---|---|---|---|
| jaren jackson jr. | 2025 | **reneg-fresh** | 2022-2025 | $35.0M | **MISMATCH — escalated** |
| julius randle | 2024 | continuation | 2022-2025 | $33.1M | ✓ |
| blake griffin | 2019 | continuation | 2017-2021 | $34.2M | ✓ |
| gordon hayward | 2019 | continuation | 2017-2020 | $32.7M | ✓ |
| kevin love | 2022 | continuation | 2019-2022 | $27.4M | ✓ |
| paul millsap | 2019 | undecidable (kept) | none | $30.5M | ✓ (no coverage; kept) |

**11/12 fresh pass; 4/6 continuation clean, Millsap undecidable-and-kept, JJJ
2025 the one mismatch** (a genuine renegotiation re-price — see §Anomalies).
Trae Young 2026 reads fresh with no instrument clash: its own [2026-2029] deal
covers it (span-fresh) and it is in the 2026 FA-list — both instruments agree.

### 2c. Part 2 — remeasured residue (date=continuation AND current filter kept)

**223 rows (was 265).** By contract year within the covering deal:

| year | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| new | 148 | 47 | 23 | 5 |
| (was) | 179 | 50 | 30 | 6 |

By season: 2019:73, 2020:30, 2021:24, 2022:17, 2023:17, 2024:20, 2025:14,
2026:28. The year-2 bucket fell 179→148 — where the span fixes bite, as the
brief predicted. **The current filter's 119 demotions are now 112 confirmed
continuation + 7 undecidable, 0 contradicted** (was 111/8/0) — precision holds.

### 2d. Part 2 — FA-list conflicts (span says earlier, but row is in the FA list)

**19 rows** (matches ISSUES #8's "19 of 265"). All KEPT by the filter (clause b).
This is the ISSUES #2 instrument clash; every one is a genuine fresh signing whose
Spotrac block anchor is unreliable.

| player | S | salary | covering span | | player | S | salary | covering span |
|---|---|---|---|---|---|---|---|---|
| al horford | 2019 | $28.0M | 2016-2019 | | montrezl harrell | 2023 | $2.0M | 2022-2023 |
| harrison barnes | 2019 | $24.1M | 2016-2019 | | edmond sumner | 2019 | $2.0M | 2018-2019 |
| draymond green | 2023 | $22.3M | 2020-2023 | | kevin knox | 2023 | $1.8M | 2022-2023 |
| duncan robinson | 2025 | $16.8M | 2021-2025 | | jamychal green | 2022 | $1.8M | 2021-2022 |
| deandre jordan | 2021 | $7.9M | 2019-2022 | | deividas sirvydis | 2021 | $1.6M | 2020-2022 |
| avery bradley | 2020 | $5.6M | 2019-2020 | | landry shamet | 2025 | $2.3M | 2022-2025 |
| justise winslow | 2021 | $3.9M | 2020-2021 | | russell westbrook | 2025 | $2.3M | 2024-2025 |
| mike muscala | 2022 | $3.5M | 2021-2022 | | eric gordon | 2025 | $2.3M | 2024-2025 |
| neemias queta | 2024 | $2.2M | 2023-2024 | | jahlil okafor | 2021 | $2.1M | 2020-2021 |
| thomas bryant | 2024 | $2.1M | 2023-2024 | | | | | |

### 2e. Part 3 — filter swap and the headline

New filter demotes **342 rows (335 by dated span + 7 by three-signal fallback)**;
frame **1,291 → 949**. Per-season: 2019:108, 2020:48, 2021:39, 2022:36, 2023:38,
2024:22, 2025:22, 2026:29. New (949) ⊆ incumbent (1,172); **common rows = 949**.
The 223 additional demotions are all mid-contract escalator years — top by salary:
Griffin 2019 (yr3), Randle 2024 (yr3), Hayward 2019 (yr3), Jrue Holiday 2024
(yr4), Love 2022 (yr4), Whiteside 2019 (yr4), Gasol 2019 (yr5) — no fresh
signing among them; heavily 2019-weighted, the recall gap the step signal missed.

**Common-row paired A1** — incumbent (1,172-frame) model vs new (949-frame)
model, both Grabit-champion, scored on the 949 common rows with one
player-fixed 5-fold split, 10 seeds, per-fold paired deltas:

| instrument | n | incumbent R² | new R² | paired Δ ± SE | t | per-fold Δ (new−inc) |
|---|---|---|---|---|---|---|
| **A1 selection (decision)** | 808 | 0.7761 | 0.7733 | **−0.0009 ± 0.0068** | **−0.13** | [+0.0105, +0.0062, −0.0183, +0.0133, −0.0162] |
| A1 pooled | 949 | 0.7788 | 0.7762 | −0.0006 ± 0.0062 | −0.09 | [+0.0052, +0.0084, −0.0182, +0.0140, −0.0123] |
| A2 2024-26 | 324 | 0.8282 | 0.8283 | +0.0013 ± 0.0051 | +0.26 | [+0.0139, +0.0045, −0.0053, +0.0084, −0.0149] |
| A2 2024-26 selection | 276 | 0.8142 | 0.8133 | +0.0001 ± 0.0055 | +0.01 | [+0.0152, +0.0005, −0.0058, +0.0075, −0.0170] |

Common-row MAE: **$3.21M → $3.21M** (unchanged). Fold sd ≈ 0.014 dwarfs the
delta — this is a null, not a signal. **Contrast: v7.7x cleared +0.0102 on this
instrument; the marginal 223 removals clear nothing.**

**B1 forward (rolling-origin 2024-26; CONTEXT ONLY, different row sets — D1):**

| frame | B1 R² | MAE | n | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| incumbent 1,172 | 0.8328 | $3.34M | 386 | 0.8528 | 0.8244 | 0.8118 |
| new 949 | 0.8191 | $3.65M | 324 | 0.8545 | 0.8097 | 0.7791 |

B1 is lower on the new frame, but the frames' test origins differ (n 386→324):
the removed 2024-26 rows are escalator years that are *easy* to predict
(prev_cap_pct ≈ current), so dropping them from the test set lowers R²'s numerator
independent of forecasting quality. Not a gate; reported per brief.

### 2f. Part 3 — C2 fixed-segment bias, common rows ($M)

Grouped by `signing_cat` over the FIXED 949 common rows; incumbent vs new OOF.

| segment | n | bias inc | bias new | Δ\|bias\| | MAE inc | MAE new |
|---|---|---|---|---|---|---|
| Bird Rights | 275 | −2.61 | −2.60 | −0.01 | 4.60 | 4.70 |
| Minimum | 243 | +2.15 | +2.02 | −0.13 | 2.26 | 2.15 |
| Unknown | 131 | +0.24 | +0.32 | +0.08 | 2.63 | 2.76 |
| MLE | 124 | +0.97 | +0.67 | **−0.30** | 2.73 | 2.61 |
| Cap Space | 78 | −0.29 | −0.45 | +0.16 | 3.33 | 3.36 |
| Early Bird | 46 | −1.37 | −1.46 | +0.08 | 2.81 | 2.73 |
| Non-Bird | 20 | +2.34 | +2.46 | +0.11 | 2.72 | 3.00 |
| Sign & Trade | 13 | −4.32 | −4.15 | −0.18 | 4.80 | 4.75 |

**No segment's |bias| grows by more than $0.3M** (max growth Cap Space +0.16);
MLE's |bias| shrinks 0.30. **C2 PASS.**

### 2g. Part 3 — the `is2019` control (ISSUES #8 verify)

Pure `is2019` dummy added to the feature set; paired ΔSel (aug − base), 10 seeds:

| frame | baseline XGB | Grabit |
|---|---|---|
| OLD 1,172 | **+0.0035 (t=2.29)** | +0.0031 (t=1.99) |
| NEW 949 | **−0.0004 (t=−0.91)** | −0.0004 (t=−0.55) |

The old-frame value reproduces the reference (+0.0032, t=2.38). On the new frame
the signal is **gone** — the season offset ISSUES #8 called "the signature of
exactly these stale prices" no longer exists once the stale rows are removed. The
2019 year-1 count falls **220 → 147**.

### 2h. Part 3 — new-frame zone sizes (do NOT compare to old $6.11M / $1.98M, D1)

| zone | n | MAE | bias |
|---|---|---|---|
| Grabit (max, ≥90% of ceiling) | 56 | $5.59M | −$5.57M |
| Floor (at CBA minimum) | 241 | $2.13M | +$2.07M |

Different rows from the old frame; reported as new absolutes only.

---

## 3. Gate verdicts

This is a filter (row-count) change, so the brief's judging criteria apply, not
the four feature gates.

| criterion | result | verdict |
|---|---|---|
| Common-row paired A1 (selection) | −0.0009, t=−0.13 | **NEUTRAL** — no positive selection signal; does not clear "accept when t>2" |
| C2 fixed-segment \|bias\| growth ≤ $0.3M | max +0.16 | **PASS** |
| `is2019` control shrinks | +0.0035 → −0.0004 | **PASS** (eliminated) |
| B1 forward (context) | 0.833 → 0.819 | reported; not a gate; D1 caveat |
| Acceptance list go/no-go | 11/12 fresh, 4/6 clean continuation, 1 undecidable-kept | **1 mismatch (JJJ 2025) escalated** |
| Filter precision (0 contradicted) | 342 demotions, 0 fresh/reneg/fa demoted | **PASS** |

**Net:** the change is CV-neutral (does not clear a positive A1 gate and does
not hurt), passes C2, and eliminates the documented `is2019` offset while
removing 223 dated-span-verified stale rows with zero contradictions. The case
for adoption is correctness at no CV cost, not a CV improvement. **Adoption is
the architect's call**, and JJJ 2025 (§4) should be resolved first.

---

## 4. Anomalies / escalations

1. **JJJ 2025 acceptance mismatch — ESCALATED (kept, not demoted).** The brief
   lists "Jaren Jackson Jr. 2025" under *stay continuation*, but on the fixed
   spans it reads **reneg-fresh** and is kept. The data: JJJ's 2021 rookie
   extension is front-loaded/declining ($28.95M in 2022 → ~$23.4M in 2025); on
   **2025-07-13 he signed a Renegotiation-and-Extend that raised his 2025-26
   salary by $11,586,605**, giving the observed $35.0M (yic=1 in the contract
   table). This is **structurally identical to Markkanen 2024 and Turner 2022**
   (both *stay-fresh* per the brief): each is a renegotiation-and-extend that
   re-prices the final year of an older deal to market. I could find no
   mechanical rule that marks Markkanen 2024 / Turner 2022 fresh while leaving
   JJJ 2025 a continuation. Per the escalation rule (instruments disagree on an
   acceptance-list item — span says continuation, the salary step 1.26 ∉ the
   escalator band says fresh — do not demote), **JJJ 2025 is kept**. This does
   not move the headline: the row is kept by both the incumbent filter (step
   fails) and the new filter (reneg carve-out), so it is a common row. The
   architect should decide whether the "stay continuation" convention for JJJ
   2025 survives the discovery that its 2025 salary is a renegotiated re-price.

2. **The two 6-year contracts are legitimate, and ISSUES #11 misstates them.**
   ISSUES #11 asserts all 328 `>5yr` contracts (incl. "2 at 6") are 10-day money
   ($41k-$176k). The 2 six-year deals are **LeBron James 2010 ($109.8M) and
   Luol Deng 2008 ($71.0M)** — real 2005-CBA-era 6-year contracts. The spec'd
   "drop > 5" removes them from `spotrac_signing_types.csv`; both start pre-2019
   and no in-window consumer reads them, and removing them costs only 2 block
   matches in `contract_signing_dates.csv` (both pre-2019). Harmless, but the
   ISSUES #11 characterization is wrong — noted in §7.

3. **Common-row A1 is neutral where ISSUES #8 expected ~+0.003.** Not too-good;
   the opposite. The +0.0032 `is2019` "worth" was the value of correctly pricing
   the *stale* rows (by removing them), which the common-row instrument cannot
   see because it scores only rows both frames keep. The benefit lands in the
   `is2019` collapse (§2g), not in common-row A1. Both are consistent.

4. **B1 lower on the new frame** — a composition/denominator artifact of a
   different (easier-rows-removed) test set, not evidence of worse forecasting;
   R² is not comparable across row sets (D1).

5. Machine note: `import pandas` under the sandboxed Bash tool ran 90s-4min per
   call on this box; PowerShell served the same in <1s. All numbers here come
   from PowerShell runs with `OMP_NUM_THREADS=6` (mirrors the signing-dates
   RESULT's note).

---

## 5. Files touched & where the branch lives

Branch **`worker/continuation-filter-v2`** at
`C:\Users\panh3\Documents\ROSE\Personal Project\BBall-worker-continuation-filter-v2`
(git worktree, cut from `master` @ `cf682ed`).

| file | change |
|---|---|
| `scripts/scrape_spotrac_players.py` | `parse_contracts`: guard the year parse against "day"; drop parsed length > 5 (ISSUES #11) |
| `scripts/parse_signing_dates.py` | `contract_spans` option-aware anchor + extension fallback +1 + `is_reneg`; new `_option_year` and `renegotiated_seasons` helpers |
| `src/model/train.py` | `_filter_continuations`: dated-span branch (demote when covering span starts earlier, unless renegotiated-fresh or in FA-list); three-signal fallback unchanged |
| `data/processed/spotrac_signing_types.csv` | regenerated (`refresh_spotrac.py --reparse-only`): 8,954 → 5,769 rows, phantoms gone |
| `data/processed/contract_signing_dates.csv` | regenerated: 2 fewer `exact` matches (the LeBron/Deng 6yr blocks) |

Reproduce (raw HTML is gitignored; a fresh worktree must point at the main
checkout's cache — I used a directory junction):

```
python scripts/refresh_spotrac.py --reparse-only
python scripts/parse_signing_dates.py
python -c "from src.model.train import *; ..."   # frame -> 949
```

Experiment scripts (not in the repo, per the worker rule) live in the session
scratchpad: `part2_measure.py`, `common_row.py`, `is2019_and_b1.py`,
`span_change_counts.py`, `zone_report.py`, `top_demotions.py`, `klay.py`.

No forbidden file touched (CLAUDE.md, METHODOLOGY.md, VERSION_HISTORY.md,
CONTEXT.md, PROJECT_BRIEF.md, evaluate_suite.py accept/reject rules).

---

## 6. Klay Thompson 2019 side-report (measure, do not fix — ISSUES #6 lane)

His 2018-19 salary **does exist upstream**: `salaries_prehistory.csv` carries
`klay thompson, 2018, $18,988,725` = **0.186 of the 2018 cap**, and
`_load_prev_season_cap_pct()` loads it correctly (returns 0.186 for
`(klay thompson, 2018)`). The failure is in the model FEATURE: `prev_cap_pct` in
`training_data_v2.csv` reads **0.037227** for Klay 2019 — a constant fill, not
his real 0.186 — because `phase3.build_contract_features` computes `prev_cap_pct`
from `training_data_v2.csv` alone (2019+), which has no 2018 row, and never
consults `salaries_prehistory.csv` the way `_load_prev_season_cap_pct` does. The
same 0.037227 fill appears on several of his rows whose true prior is present
(2020-2023, 2025-2026), so the feature is broken for more than the 2019 boundary.
The fix (ISSUES #6 lane) is to have `build_contract_features` fall back to the
prehistory table for pre-2019 priors, exactly as the training helper already does.

---

## 7. Proposed commit message (architect edits and lands; version unclaimed)

> Continuation filter v2: dated-span demotion (ISSUES #8 + #11)
>
> Fix the 10-day parse bug (10 yr(s) read as a 10-YEAR contract): reject parsed
> length > 5 and never read a year from a "day" field. spotrac_signing_types.csv
> 8,954 -> 5,769 rows; contract_years.max() 5, no start before 1990; the
> continuation filter still drops the same 119, so no phantom span was
> load-bearing. 144/146 phantom-contract players independently class as 10-day.
>
> Fix contract_spans: option-final extensions were anchored a season early
> (Spotrac's fa_year is the option-decision summer) -> slide the span onto the
> option season when option_year exceeds the nominal end (49 spans, 14 change
> start); unmatched extensions start signing_season+1 (24); a renegotiation-and-
> extend marks its signing season fresh (6). Embiid 2023 / Doncic 2022 / Gobert
> 2021 / Butler 2023 now read year 1; Trae 2021 and Randle 2021 extensions are
> correctly left unshifted. ISSUES #9 invariant preserved.
>
> Swap _filter_continuations to demote a row outright when a dated contract
> covers it and starts earlier, unless the season is renegotiated-fresh or the
> row is in that season's FA-signings list; three-signal consensus stays the
> fallback for undated rows. Frame 1,172 -> 949 (342 demoted, 0 contradicted).
> Common-row paired A1 selection -0.0009 (t=-0.13, neutral); MAE unchanged; C2
> max |bias| growth +0.16M; the is2019 dummy collapses +0.0035 -> -0.0004; 2019
> year-1 count 220 -> 147.
>
> Open: JJJ 2025 reads reneg-fresh, not the "stay continuation" the brief
> expects, because its 2025 salary is a genuine renegotiation re-price
> (structurally identical to Markkanen 2024 / Turner 2022) — kept, escalated.

Version number deliberately unclaimed — the assignment is the architect's.

---

## 8. ISSUES.md additions

Applied to `ISSUES.md` on this branch:

- **#8 / #11** — annotated as addressed on `worker/continuation-filter-v2`
  pending the architect's landing (not deleted; the fix is not on master).
- **new #13** — `prev_cap_pct` feature ignores `salaries_prehistory.csv`
  (Klay 2019 fill; §6).
- **new #14** — the JJJ 2025 renegotiation-vs-continuation convention question
  (§4.1), for the architect to adjudicate.
- **#11 correction** — the "2 at 6" are legitimate LeBron/Deng 6yr deals, not
  10-day money (§4.2).
