# Open issues

Known problems that are real but were out of scope when found. Each entry is
written to be actionable without the conversation that produced it: what is
wrong, how to see it for yourself, what to do, and how to know you fixed it.

**If you find something you are not fixing right now, add it here** rather than
leaving it in a chat log. Delete an entry when it is fixed — this file is a
worklist, not a changelog. `VERSION_HISTORY.md` is where fixes get recorded.

Ordered roughly by how much damage each one does.

---

## 1. The published Signing Board shows in-sample residuals

**Severity**: high — it is a public accuracy claim, and it is flattering by 41%.

`scripts/export_web.py` fits Grabit on the 1,556 training rows and then scores
those same rows. Every residual the Signing Board displays is in-sample, on a
page whose own copy calls it "a genuine read on accuracy".

| | MAE |
|---|---|
| What the board shows (in-sample) | $2.48M |
| Cross-validated, 2024–26 | $3.60M |
| Cross-validated, all seasons | $4.16M |

This is also why the site's mechanism biases disagree with `METHODOLOGY.md` —
Bird Rights reads −$2.17M on the board against −$2.78M in the docs, on the same
n=58. The docs are right.

**Reproduce**:

```bash
python -c "
import pandas as pd, json
d = pd.read_csv('outputs/web/valuations_export.csv')
sb = d[(d.year_in_contract == 1) & (~d.is_rookie_scale)]
m = json.load(open('outputs/models/grabit_results.json'))
print(f'in-sample MAE  \${(sb.pred_salary - sb.actual_salary).abs().mean() / 1e6:.2f}M')
print(f'cross-val MAE  \${m[\"cv_mae_mean\"] * 166e6 / 1e6:.2f}M')
"
```

**Fix**: `train_grabit()` in `src/model/train.py` already computes exactly what
is needed — `oof_pred`, the out-of-fold prediction for every training row — and
then throws it away. Return it alongside the fitted model, and have
`build_frame()` prefer the out-of-fold value for any row in the training set,
falling back to the fitted model only for rows the model never saw (escalator
years, rookie-scale seasons — the Value Board's extra rows).

Note the two boards then rest on different predictions by design, which is
correct: the Signing Board asks "how well does this generalise", the Value Board
asks "what is this player worth", and only the first has a holdout answer.
`components/nba/Explorer.tsx` copy should say so.

**Verify**: Signing Board MAE lands near the cross-validated figure rather than
$2.48M, and mechanism biases match `METHODOLOGY.md`.

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

**Verify** (state at v7.8x): evaluation frame 1,172 rows; 2019 year-1 count 220
against 255 before; `check_caps.py` hit rates for 2019-2020 risen.

---

## 3. Site and docs quote CV R² from different runs

**Severity**: low — 0.0002 apart, but they are the same headline number.

`PROJECT_BRIEF.md` gives Grabit CV R² as 0.7581 (10-seed averaged);
`outputs/models/grabit_results.json`, which the site reads through
`export_web.py`, holds 0.7583 from a single seed. Both are defensible, but a
reader comparing the page against the brief sees two answers.

**Fix**: decide which is canonical. If it is the 10-seed average, have
`train.py` write that into `grabit_results.json` so the export inherits it
automatically.

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
against `data/processed/training_data.csv`. The valuation board renders these
as "—". Neither field is a model feature, so this affects presentation and the
team filter only.

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
