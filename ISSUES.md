# Open issues

Known problems that are real but were out of scope when found. Each entry is
written to be actionable without the conversation that produced it: what is
wrong, how to see it for yourself, what to do, and how to know you fixed it.

**If you find something you are not fixing right now, add it here** rather than
leaving it in a chat log. Delete an entry when it is fixed — this file is a
worklist, not a changelog. `VERSION_HISTORY.md` is where fixes get recorded.

Ordered roughly by how much damage each one does.

---

## 1. The published site is seven versions stale and cannot be republished as-is

**Severity**: high — it is a public accuracy claim, it is flattering, and the
one command that would refresh it is now broken in three places.

`outputs/web/valuations_export.csv` was written 2026-07-22 with 3,113 rows and a
1,556-row Signing Board: that is the **v7.1x** row set, before the prorated,
mislabel and continuation filters. `outputs/models/grabit_results.json` holds
`Grabit v3, n=1297, cv_r2 0.7586` — the **v7.2x-v7.5x** era. The docs describe
v7.8x (n=1,172, two-sided censoring). Every number on the site disagrees with
every number in `METHODOLOGY.md`.

### 1a. The residuals it shows are in-sample

`scripts/export_web.py` fits Grabit on the training rows and then scores those
same rows, on a page whose own copy calls it "a genuine read on accuracy".

| | MAE |
|---|---|
| What the board shows (in-sample) | $2.48M |
| Cross-validated, v7.8x pooled | **$3.07M** |
| Cross-validated, v7.8x 2024-26 | **$3.19M** |

**Fix**: `train_grabit()` already computes `oof_pred`, the out-of-fold prediction
for every training row, and throws it away. Return it, and have `build_frame()`
prefer the out-of-fold value for any row in the training set, falling back to the
fitted model only for rows the model never saw (escalator years, rookie-scale
seasons — the Value Board's extra rows).

The two boards then rest on different predictions *by design*, which is correct:
the Signing Board asks "how well does this generalise", the Value Board asks
"what is this player worth", and only the first has a holdout answer.
`components/nba/Explorer.tsx` copy should say so.

### 1b. `export_web.py` has drifted from the model it exports

Three concrete breakages, all introduced by v7.6x-v7.8x:

- **Filter chain** (`_training_medians`, line ~134) runs
  `_filter_prorated(_filter_rookie_scale(_filter_year1(df)))` and stops there.
  Its own docstring says the chain "must stay identical to the one inside
  `train_grabit`"; it is missing `_filter_mislabeled_year1` and
  `_filter_continuations`, so imputation medians come from 1,297 rows while the
  model is fit on 1,172.
- **Stage 2** (`build_frame`, line ~157) is `np.minimum(latent, max_elig)` — the
  ceiling only. v7.8x's Stage 2 is `clip(latent, floor_pct, max_eligible_pct)`,
  so at-floor players would publish with un-clipped latents. `is_capped`
  (line ~205) likewise only detects the ceiling; a floor-clipped row needs its
  own flag for the site to explain what it is looking at.
- **Label** (line ~473) prints "Grabit v3" unconditionally.

**The `canon` guard is working as designed** and will refuse to publish: it
compares the fresh fit's CV R² against `grabit_results.json` and exits on a
mismatch >1e-6. Re-run `src/model/train.py` after fixing the above, so the
quoted metrics and the table describe the same model.

**Verify**: Signing Board MAE lands near the cross-validated figure rather than
$2.48M; mechanism biases match `METHODOLOGY.md` (Bird Rights −$2.50M on n=309);
`meta.json` reports n=1,172 and a two-sided model.

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

**Severity**: low on its own; it becomes the visible symptom of #1 whenever the
site is republished.

`src/model/train.py` writes `grabit_results.json` from a **single-seed** run;
`METHODOLOGY.md`, `PROJECT_BRIEF.md` and `evaluation_suite.json` quote the
**10-seed** average. Both are defensible numbers, but they are reported under
the same name, and `export_web.py` propagates the single-seed one to the site
through `meta.json`.

**Fix**: pick one as canonical — the 10-seed average is the one every document
uses — and have `train.py` write that into `grabit_results.json` so the export
inherits it automatically. Note this interacts with the `canon` guard in
`export_web.py`, which compares its own single fit against that file; if the
file holds a 10-seed mean, the guard needs to compare against a 10-seed mean too
or it will always trip.

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

  The signing-date scrape now has a fourth quantified payoff. The exit-structure
  ablation (2026-07-23) found the length of the expiring contract carries real
  signal — +0.0032 paired (t=2.32) as a pure increment over the coverage
  artifact — but only 54% of evaluation rows can see their expiring contract in
  the current Spotrac blocks, and every deployable fill for the other 46%
  either reopens the collection-artifact channel (sentinel) or drowns the
  signal in a poisoned bucket (mode fill scored +0.0002). Transaction-level
  signing dates would push coverage toward ~95%, where the unknown share is too
  small to matter and the locked +0.003 becomes collectible. RFA status, by
  contrast, tested empty once the artifact was removed (+0.0004), and option
  structure is unmeasurable historically — Spotrac annotates options on current
  contracts only (has_po is 0.000 for every fa_year before 2023 and 0.35-0.50
  for 2028+).
- **`floor_pct`** is the median pay of at-floor rows per (season, experience
  bucket) — a recovery of the veteran-minimum scale from the data's own mass
  points, not the published scale. Buckets with few at-floor rows fall back to
  the season minimum. Accurate enough that the Stage-2 clip lands within
  $0.05-0.09M of observed pay, but a published scale would be exact.

**Verify**: `python -c "from src.model.train import *; ..."` — over-cap rows stay
0, and at-floor rows in the sub-2% predicted band keep bias under $0.10M.

---

## 7. Sigma and both gates were tuned on a different row set

**Severity**: low — the settings still pass their zone tests, so this is
opportunity rather than damage.

`sigma = 0.02` dates from a 1,487-row training set with 73 rows in the max zone.
The zone is now 57 rows of 1,172, and a second censoring side exists that did
not when sigma was chosen. The right gate (0.55) is equally old; only the left
gate (k = 2.0) was screened on the current data, over {1.5, 2.0, 3.0}.

**Fix**: re-sweep sigma and both gates judged on **zone** MAE rather than pooled
R², with the pooled selection-pool delta as a no-regression guard. The screening
harness from the v7.8x experiment is the pattern to copy.

**Verify**: whichever settings win, both zone MAEs stay at or below $6.11M
(max) and $1.98M (floor), and the selection-pool paired delta does not go
negative.

---

## 8. Spotrac 10-day contracts are parsed as 10-YEAR contracts

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
`python scripts/refresh_spotrac.py --reparse-only` (no network).

**Verify**: `contract_years.max() <= 5` over the whole table; no contract has
`contract_start` before 1990; the continuation filter still drops 119 rows with
the same per-season split (2019: 35 … 2026: 1) — if that count moves, a phantom
span *was* load-bearing and the change needs a common-row A1 delta before it
lands.

---

## 9. AAV is unavailable for half the evaluation frame, non-randomly

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
signed-FA page. Transaction-level signing dates (already wanted by #6) would
raise coverage the most.
