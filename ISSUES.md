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

## 2. Six pre-2022 escalator years still wear a year-1 label

**Severity**: medium — corrupts the definition of a "fresh signing" for six
training rows. (Residual of a larger entry: the ceiling side of the original
"thirteen rows above their max" issue was closed in v7.4x/v7.5x — award trigger
now accepts `s-1`, hand-curated `early_supermax.csv` covers extensions signed
two summers early (a blanket s-3 lookback was tried first and wrongly
un-censored Klay 2019 and Fox 2026 — do not reintroduce it), and a
1.08 x prior-pay floor with `salaries_prehistory.csv` anchors the 2019 boundary.
`prev_cap_pct` staleness was fixed by the v7.3x rebuild: spot-check shows every
derivable row matches the corrected-cap value, not the stale one.)

What remains: Stephen Curry 2019, John Wall 2020, Chris Paul 2019, Russell
Westbrook 2019, Andrew Wiggins 2019, CJ McCollum 2019 are later years of older
deals labelled year-1 — contracts signed before 2019 have their first
*observed* season tagged year 1, because the lost structure script only saw
2019+ data. They no longer break the ceiling (the 1.08 floor covers them), but
they still sit in training as if they were fresh market prices. This is also
why `scripts/check_caps.py` reports low hit rates for 2019-2020.

**Reproduce**: compare each row's salary to its *tier-only* ceiling (call
`_compute_max_eligible` and recompute `base` without the prior-pay floor); the
six rows exceed it while no genuine fresh signing does.

**Fix**: `salaries_prehistory.csv` now holds 2016-2018 pay for 155 players —
enough to run the escalator-chain detection backwards across the 2019 boundary
for exactly these players and demote mislabeled rows. Alternatively demote any
year-1 row whose salary exceeds its tier-only ceiling by more than rounding;
that condition is proof of mislabelling, not a judgement call.

**Verify**: the six rows leave the year-1 training set; `check_caps.py` hit
rates for 2019-2020 rise; A1 moves little (n=6).

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

## 6. Docs lag three landed decisions

**Severity**: medium — METHODOLOGY and VERSION_HISTORY describe a model two
versions behind the code, and one confirmed architecture decision exists only
in a chat log until it lands here.

What needs writing, in priority order:

1. **Two-stage semantics, as confirmed 2026-07-23**: Stage 1 predicts value
   under *default parameters* (today: implicit training-set averages over
   mechanism/market context); Stage 2 adjusts for *told parameters* (today:
   exactly one — the legal ceiling `max_eligible_pct`). The C2
   mechanism-bias table measures precisely the context that Stage 1 averages
   over and Stage 2 does not yet condition on; any parameter promoted from
   "averaged" to "told" should shrink its C2 bias, which is the natural
   acceptance test. Belongs in METHODOLOGY (model section) and CONTEXT.md
   (vocabulary: consider naming Stage-1 output "reference value").
2. **VERSION_HISTORY entries for v7.3x-v7.5x** — the annotated git tags carry
   the one-liners and A1/A2/B1 for each.
3. **Grabit's keep/drop rule** — judged on its zone (rows paid >= 90% of their
   own ceiling: MAE, n=65-67, printed by the suite), not the pooled paired
   delta, which mixed a real zone effect with dilution and, before v7.4x, with
   a ceiling bug. Replaces the "Grabit Impact on Max Contracts" story in
   METHODOLOGY.
4. **Prediction-timepoint convention** — features must be knowable at
   market open (July 1). This is why exit-mechanism features use option
   *structure* (knowable at signing) rather than option *decisions*
   (knowable only at market open), and why FA supply must be computed from
   contract expirations rather than realized FA lists.

**Verify**: a reader of METHODOLOGY alone can reproduce the current suite
output without visiting this file or the git log.
