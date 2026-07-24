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
are kept outright — precision over recall; Millsap 2019 is the canonical kept
suspect. Expanding `salaries_prehistory.csv` or the player-page transactions
cache converts more of them into decidable rows.

**Verify** (state at v7.9x): evaluation frame 949 rows; 2019 year-1 count 147
(was 220); the filter prints "dropped 342 rows (335 by dated span, 7 by
three-signal consensus)".

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

## 16. `salaries_prehistory.csv` misses ~6 of the 2019 eval rows

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

## 15. Record v7.9x and its conventions in VERSION_HISTORY / METHODOLOGY

**Severity**: low — docs lane only; the decisions are made and the code is on
master.

- **VERSION_HISTORY**: v7.9x = continuation filter v2 + the 10-day parse fix
  (`spotrac_signing_types.csv` 8,954 → 5,769 rows). Frame 1,172 → 949 (342
  demoted, 0 contradicted). Common-row paired A1 −0.0009 (t=−0.13) — neutral
  by construction, since the instrument scores only rows both frames keep and
  is blind to the benefit of removals; the pre-registered success criterion
  was the `is2019` control, +0.0035 (t=2.29) → −0.0004 (t=−0.91). C2 max
  |bias| growth +0.16M. New-frame A1/A2/B1 are in the refreshed
  `evaluation_suite.json`.
- **METHODOLOGY**, three items:
  1. The **renegotiation convention**, adjudicated 2026-07-24: a
     renegotiation-and-extend re-prices its signing season to market, so that
     season is FRESH even though an older span covers it. Rules all six pairs
     the data contains — Turner 2022, Sabonis 2023, Clarkson 2023, Isaac 2024,
     Markkanen 2024, and JJJ 2025 (whose 2025-26 salary was raised $11.6M on
     2025-07-13; the one acceptance-list mismatch, resolved this way).
  2. The **span rules**: Spotrac's `fa_year` on an option-final deal is the
     option-decision summer, so the span is `[fa_year − years + 1, fa_year]`;
     an unmatched extension starts paying at `signing_season + 1`; 120 block
     anchors (6.1%) predate their own signing and are date-resolved. A block
     anchor alone is never a trustworthy span.
  3. The **protocol lesson**: a pure-removal change is judged on its
     pre-registered contamination signal plus guardrails (C2, common-row
     neutrality), not on common-row A1 improvement — that instrument cannot
     see removals by construction.

