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

**Severity**: low — everything decidable is gone as of v7.9x; what remains is
kept on purpose.

Closed in three steps. v7.6x: six rows paid above their *tier* ceiling
(`_filter_mislabeled_year1`). v7.7x: 119 rows on a three-signal consensus
(+0.0102 common-row A1). v7.9x: the dated-span branch — option-aware anchors,
extension fallback +1, renegotiation carve-out — demotes 342 rows total (335
by dated span, 7 by the consensus fallback), frame 1,291 → 949, zero demotions
contradicted by any instrument. The `is2019` season offset that quantified the
remaining contamination (+0.0035, t=2.29 at v7.8x) measures −0.0004 (t=−0.91)
on the new frame: the stale class is gone from where it could be seen.

**A warning that must outlive this entry**: a BLOCK-anchor span alone is NOT
sufficient. A span-only rule once deleted Brunson 2022, VanVleet 2023 and
Jimmy Butler 2019 — all genuine fresh signings; Spotrac's Free-Agent anchor is
unreliable for contracts later superseded or option-final. The v7.9x dated
branch is span-based but keeps the FA-signings-list veto for exactly that
reason: 19 rows whose dated span reads continuation but who sit in that
season's FA list are KEPT (Horford 2019, Draymond 2023, …). **Do not remove
the veto.**

What remains, deliberately: rows no dated contract covers fall back to the
three-signal consensus, and rows whose last-season pay is also unobservable
are kept outright — precision over recall; Millsap 2019 was the canonical kept
suspect. Expanding `salaries_prehistory.csv` or the player-page transactions
cache converts more of them into decidable rows — and did: the 2018 team-salary
scrape (2026-07-24) made five of them decidable and demoted (Millsap, Snell,
Galloway, Okafor, Ferrell — all escalator continuations, none in the FA list),
frame 949 → 944.

**Verify** (state after the prev_cap_pct landing): evaluation frame 944 rows;
2019 year-1 count 142; the filter prints "dropped 347 rows (335 by dated span,
12 by three-signal consensus)".

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

- **`designated_ineligible.csv`** is the inverse curated list (added v7.12x,
  ISSUES #19): rows the award path wrongly bumps to a 35%/30% designated ceiling
  because it fires on All-NBA + experience alone and cannot see that the player
  changed teams (Kawhi 2019), was acquired on a veteran deal (AD 2020), or
  signed the 25% base of a rookie-extension whose Rose escalator lands in a later
  year (KAT 2019, Tatum 2021). 12 rows; each hand-verified. Same team-continuity
  data gap as early_supermax — a signing-date team-match test would replace both
  lists. Maintenance: a new such deal each summer needs a row.

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

## 17. `salaries_prehistory.csv` misses ~6 of the 2019 eval rows

**Severity**: low — the affected rows carry negligible prior pay, so their
`prev_cap_pct` slot-fill fallback is nearly right anyway.

After the 2018-19 team-salary scrape (`scripts/scrape_2018_salaries.py`), 2018
prior coverage of the season-2019 eval rows reached 136/142. The remaining 6 are
two-way / minimum players whose salary is absent from the BBRef `salaries2` team
table (Alex Caruso, Shake Milton, Wes Iwundu, Wenyen Gabriel, Amile Jefferson),
plus one **name rename**: **Enes Kanter → Enes Freedom**. His 2018-19 pay is on
the page under "Enes Kanter" (`norm` → `enes kanter`), but the training data
carries `enes freedom`, so the join misses it.

**Fix**: a small name-alias map applied in `scrape_2018_salaries.py` /
`_load_prev_season_cap_pct` (kanter→freedom, and any other post-2019 renames)
would recover the vet row; the two-way players need a two-way salary source, not
the team salary table. Low priority — none of the six moves a model feature
materially.

**Verify**: `_load_prev_season_cap_pct()` returns a value for
`("enes freedom", 2018)`.

---

## 18. `awards_full.csv` name join drops footnote-marked stars from award_score_cum

**Severity**: low — real data defect, but fixing it does NOT help the model
(measured slightly negative), so it is documentation, not a pending fix.

90 of 943 rows in `data/raw/raw_external/awards_full.csv` carry footnote junk in
`player_name_norm` — trailing `^` (the dominant marker), `§`, `†`, the Unicode
replacement char, and concatenated tokens (`bam adebayost1`, `ben simmonscovid2`).
`scripts/build_external_features.norm` strips accents and lowercases but does
**not** remove these, so `norm("Kevin Durant^") == "kevin durant^"` and the join
to the training names misses. The affected rows are exactly the high-value
players: **all 13 Rookie-of-the-Year winners** (Wiggins, Towns, Luka, Ja, LaMelo,
Barnes, Banchero, Wembanyama, …) and MVP / All-NBA seasons for Durant, Curry,
Giannis, Jokić, Harden, Westbrook. So `award_score_cum` and `all_nba_cum`
silently under-credit the stars whose award history matters most.

**See it:**
```
python -c "import pandas as pd; aw=pd.read_csv('data/raw/raw_external/awards_full.csv'); \
print(aw[aw.player_name_norm.str.contains(r'[^a-z .\'-]', regex=True)].award.value_counts())"
```
Every ROY row appears; `norm('Kevin Durant^')` still ends in `^`.

**What to do (if ever):** strip the junk with
`re.sub(r"[^a-z .'-]", "", name)` inside `norm` (or a dedicated award-name
cleaner) before the award merge. `scripts/feature_batch._clean_award_name` is a
working implementation. **But do not expect a model gain:** the 2026-07-25
feature batch measured the name-cleaned `award_score_cum` (control arm
`e_cleanonly`) at paired ΔSel −0.0042 (A2 −0.0052) against the incumbent — the
recovered superstar award mass is redundant with `darko`/`prev_cap_pct` and the
players are ceiling-pinned, so it adds variance without lift. Fix it for
correctness and any future award-based diagnostic, not for CV.

**Verify a fix:** `norm("Kevin Durant^") == "kevin durant"` and
`award_score_cum > 0` for luka doncic / ja morant in their post-rookie seasons.



---

## 20. The experience fallback (`age − 19`) over-tiers players with no draft year

**Severity**: moderate — same damage shape as the fixed #19 (a real max wears
a ceiling one tier too high, so `is_max_contract` misses it and the Stage-2
clip sits above the truth), on a different code path.

`_compute_max_eligible` derives service years as `season − draft_year`, and
where the draft table has no entry it falls back to `age − 19`
([train.py:483](src/model/train.py:483)). That fallback fires on **223 of 944
evaluation rows (24%)** and systematically overstates service for anyone who
entered late or undrafted — every extra year pushes toward the 7-9 (30%) and
10+ (35%) brackets.

Two rows are provably mis-tiered by it, and both signed **exactly at a tier**,
which is the signature of a max:

| row | pay | true tier | tier granted | real service | fallback said |
|---|---|---|---|---|---|
| austin reaves 2026 | **25.000%** of cap | 25% | 30% | 5 (undrafted 2021) | 8 |
| jimmy butler 2019 | **30.000%** of cap | 30% | 35% | 8 (drafted 2011) | 10 |

Both currently read `is_max_contract = False` at ~83-86% of their inflated
ceilings. Reaves is also the second-largest collateral row in the route-mixture
phase-3 re-run (+$9.71M push damage) purely because of this label.

**Reproduce**:

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.model.evaluate_suite import load_evaluation_frame; d,_=load_evaluation_frame(); m=d[d.player_name_norm.isin(['austin reaves','jimmy butler'])&d.season.isin([2026,2019])]; print(m[['player_name_norm','season','cap_pct','tier_ceiling_pct','is_max_contract']])"
```

**Fix**: give the ceiling rule a real service-year source instead of the age
proxy — first NBA season per player, derivable from the cached BBRef player
pages already used by `scripts/height.py` (or from the earliest season in
`salaries.csv` + `salaries_prehistory.csv` as an offline approximation, which
covers 2016+ and is exact for anyone whose debut is inside that window). Keep
`age − 19` only as a last resort and log how many rows use it. A curated
two-row patch (mirroring `designated_ineligible.csv`) is the cheap stopgap if
the service-year source is deferred, but the systematic fix is preferred —
24% of rows currently rest on the proxy.

**Verify**: Reaves 2026 `tier_ceiling_pct == 0.25` and `is_max_contract` True;
Butler 2019 `tier_ceiling_pct == 0.30` and True; the max zone grows 68 → 70;
no row's ceiling falls below its own pay (`over-tier count == 0`); frame stays
944.
