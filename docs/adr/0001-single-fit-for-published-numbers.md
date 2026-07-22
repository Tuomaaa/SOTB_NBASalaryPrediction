# Every published number comes from one model fit

`outputs/predictions/` accumulated a dozen CSVs across model versions, and they
disagree: for the 2026 season, `predictions_2026_shap.csv` and
`valuation_full_dataset.csv` differ by more than $5M on 8% of shared players and
by $13.6M at worst, while `valuation_2026_grabit_v3.csv` covers only 55 players.
Assembling the public site from whichever files happened to have the needed
columns would have put a SHAP waterfall next to a predicted salary it does not
sum to. So `scripts/export_web.py` refits Grabit itself and derives the table,
the attributions, and the charts from that single fitted model — it deliberately
reads none of the prediction CSVs sitting next to it.

## Consequences

The published figures move whenever the export is re-run, which is correct but
means the site's numbers are not pinned to any committed CSV. `outputs/web/`
holds a snapshot of each export so a published figure can be traced back.

Refitting costs seconds on 1,556 training rows, so the convenience argument for
reusing the CSVs was never worth much. The expensive path — 10-seed × 5-fold CV
for the headline metrics — stays in `train.py` and is read from
`outputs/models/grabit_results.json`; the export reproduces the same fit and
asserts its CV R² matches.

The 2026-07-22 refresh is a worked example of why the snapshot matters: the
prediction CSVs in `outputs/predictions/` were computed against salary caps for
2025 and 2026 that were later found to be wrong, so every dollar figure in them
is inflated by roughly 9% for those two seasons. They are historical artifacts,
not a source.
