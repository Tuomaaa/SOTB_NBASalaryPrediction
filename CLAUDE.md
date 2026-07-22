# CLAUDE.md — NBA Free Agent Valuation Model

## Project Overview

Build an open-source, reproducible NBA free agent valuation model inspired by Hollinger's BORD$. The model predicts a player's market value as **% of salary cap**, using publicly available data and open advanced metrics as the base rating.

The key differentiator: unlike BORD$, this model is fully transparent, reproducible, and incorporates CBA structural constraints as explicit features.

## Architecture

```
nba-valuation/
├── CLAUDE.md                # this file — instructions and conventions
├── AGENTS.md                # pointer to this file for agents that look for it
├── CONTEXT.md               # domain glossary — the vocabulary code and docs share
├── METHODOLOGY.md           # feature definitions, model math, ablations, limits
├── VERSION_HISTORY.md       # v1.0 → v7.1x, CV R² at each step
├── PROJECT_BRIEF.md         # outward-facing summary
├── docs/adr/                # architecture decisions and rejected alternatives
├── config.py                # seasons, cap values, CBA params  ← caps are load-bearing
├── src/
│   ├── scraping/
│   │   ├── contracts.py     # Basketball Reference team + player salary pages
│   │   ├── stats.py         # BBRef per-season advanced stats
│   │   ├── advanced.py      # nbarapm.com: DARKO DPM, LEBRON, RAPM
│   │   ├── height.py        # BBRef player index → height in inches
│   │   ├── availability.py  # weighted GP% from the games column
│   │   └── utils.py         # cached, rate-limited fetch via Playwright
│   ├── features/
│   │   ├── base_rating.py   # z-score the three impact metrics within season
│   │   ├── age_curve.py     # age, age²
│   │   ├── availability.py  # 3-year weighted GP%
│   │   ├── cba_constraints.py  # CBA era flag
│   │   └── build_dataset.py # stage 1 of the training-data rebuild
│   └── model/
│       ├── train.py         # Ridge / XGBoost / Grabit + the year-1 and rookie filters
│       ├── evaluate_suite.py  # the four-layer evaluation protocol (see below)
│       ├── evaluate.py      # residual plots
│       └── predict.py       # inference on upcoming free agents
├── scripts/
│   ├── build_external_features.py  # stage 2: awards, draft, injuries, team value
│   ├── phase3.py            # stage 3 lives here: build_contract_features → prev_cap_pct
│   ├── check_caps.py        # asserts every season's cap reconciles with max contracts
│   ├── diagnostics.py       # residual analysis, SHAP, signing-mechanism slices
│   ├── export_web.py        # refits and writes the portfolio site's data
│   └── scrape_*.py          # one-off scrapers for awards, Spotrac, missing salaries
├── data/
│   ├── raw/                 # scraped HTML cache + hand-curated CSVs (mostly gitignored)
│   └── processed/           # cleaned, merged datasets
├── versions/v1/             # frozen snapshot so old version numbers stay reproducible
└── outputs/
    ├── models/              # metrics JSON (gitignored)
    ├── predictions/         # historical prediction CSVs — see ADR 0001, do not reuse
    ├── diagnostics/         # residual tables and plots
    └── web/                 # export snapshots
```

**Rebuilding the training data** runs three stages in order, and there is no
single entry point that chains them:

```
src/features/build_dataset.py   →  base merge
scripts/build_external_features.py  →  awards, draft, injuries, team value
scripts/phase3.py::build_contract_features  →  prev_cap_pct
```

The script that produced `contract_structure_v2.csv` was never committed. A
reconstruction from CBA escalator ratios only reaches 88% agreement on the
year-1 flag, which is far too low to regenerate history without invalidating
every published version number — so that table is extended incrementally
(unchanged rows keep their assignment) rather than recomputed.

## Data Sources (all public)

| Source | Data | Format |
|--------|------|--------|
| Basketball Reference | salary by season, age, games played, height | HTML scrape (cached) |
| nbarapm.com | DARKO DPM, LEBRON, RAPM, usage, box-score rates | Playwright + POST |
| Spotrac | signing mechanism, contract years, total value, AAV | HTML scrape (cached) |
| Manual reference | awards, draft position, team value | hand-curated CSV in `data/raw/raw_external/` |
| `config.py` | salary cap by season, CBA era boundary | hand-maintained |

Sources considered and not used: Dunks & Threes (EPM), NBA.com tracking data.
The three impact metrics from nbarapm.com already cover the performance signal —
see the ablation table in METHODOLOGY.md.

**Salary cap values are load-bearing.** `cap_pct` is the model target, so a wrong
cap silently rescales the target for an entire season. This has already happened
once: the 2025 and 2026 caps sat at stale pre-media-deal projections, inflating
those seasons' targets by roughly 9%. `scripts/check_caps.py` now asserts every
season reconciles against max contracts landing on exactly 25/30/35% of the
configured cap; run it after touching `CAP_BY_SEASON`.

## Target Variable

`cap_pct` = annual salary / salary cap for that season. CONTEXT.md calls this
**Cap Percentage**; use that name in prose and `cap_pct` in code.

Training uses **Year-1 Contracts only**. Escalator Years are CBA-mandated raises
on a price agreed years earlier, and Rookie-Scale Contracts are slotted by draft
position — neither carries market information. Scoring, by contrast, runs over
every row, because a Contract Surplus on an escalator year is a real statement
about a team's books even though it is not a Signing Residual.

## Feature Set

14 features, listed with definitions in METHODOLOGY.md. Over 20 further
candidates were tested and rejected, each with its ΔCV R² recorded in the same
file — consult that table before proposing a feature, since several obvious
ideas (team cap space, playoff performance, agent portfolio) are already there.

Two rules the ablation table encodes:

- **Never feed the model anything derived from the target.** `is_vet_min`,
  `is_mle_range`, and `is_rookie_scale` produced large gains in Phase 2 and were
  all leakage; they were removed in v4.0.
- **Signing Mechanism is a diagnostic label, not a feature.** It is partly
  determined by the contract itself. Feeding the model a fold-honest
  `P(mechanism | x)` was tested and *hurt* (−0.0073), because the tree model
  already extracts everything the features say about mechanism.

## Modeling Strategy

Current model is the two-stage Grabit pipeline described in METHODOLOGY.md.
Ridge remains in `train.py` as a reference point.

Escalating model complexity requires a paired CV improvement, not a hunch. The
hyperparameters have been grid-searched twice and the model is **not**
underfitting — deeper trees, higher learning rate, and looser `min_child_weight`
all score worse. Extra structure has to justify itself against that.

## Coding Conventions

- Python 3.10+
- `pandas` for data, `scikit-learn` for modeling, `xgboost` for GBT, `shap` for attributions
- `matplotlib` / `seaborn` for viz
- Type hints encouraged but not mandatory
- Docstrings for all functions in `src/`
- Scraping: `BeautifulSoup` over Playwright-fetched HTML, 3s between live requests, everything cached under `data/raw/html_cache/`
- Dollar amounts stored as `cap_pct` (float 0-1), never raw dollars. Dollars are a display unit only
- Use CONTEXT.md's vocabulary in prose. In particular keep **Signing Residual** (model accuracy, Year-1 only) distinct from **Contract Surplus** (team outcome, any year) — the arithmetic is identical and the meanings are not

**Every entry point under `src/` and `scripts/` starts with**

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # ".parent.parent" under scripts/
```

Running `python foo.py` puts only the script's own directory on `sys.path`, not
the working directory, so `from config import ...` will not resolve without it.
Keep the bootstrap when adding a new entry point.

**Do not rebind `sys.stdout`.** Use `sys.stdout.reconfigure(encoding="utf-8")`.
Assigning `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)` at module level
leaves the previous wrapper to be garbage-collected, which closes the underlying
buffer for any process that imports the module.

## Evaluation Protocol

`src/model/evaluate_suite.py` implements four layers. They answer different
questions and must not be mixed or substituted for one another.

| Layer | What it is | What it is for |
|-------|-----------|----------------|
| **A** selection | pooled GroupKFold CV, and its 2024-26 subset | every accept/reject decision |
| **B** forecasting | rolling-origin, train on all seasons < T, for T in 2024-2026 | what `predict.py` will actually achieve |
| **C** integrity | calibration and per-segment bias | catching a change that helps the average and hurts a segment |
| **D** guards | fixed eval set, baseline ladder, locked confirmation split | comparability |

Rules that are easy to get wrong:

- **Pooled CV is not leaking.** The estimand is the market's pricing function, a
  structural quantity, so using later seasons to estimate it is efficient rather
  than optimistic. Player grouping blocks the leakage that does matter. Layer B
  exists because `predict.py` genuinely forecasts, not because layer A is dishonest.
- **Report deltas paired by fold.** Fold sd is ~0.046; seed sd is ~0.001.
  Comparing two independently-reported means throws away nearly all the power.
- **Bin calibration by predicted value, never by the target.** Binning residuals
  by actual salary produces a steep monotone bias gradient even when calibration
  is perfect. Earlier versions of METHODOLOGY.md reported exactly that artifact
  as a finding.
- **Fix the evaluation set when the training filter changes.** R²'s denominator
  moves with the dataset, so R² across different row sets is not comparable.
- **The confirmation split is 15% of players, held out of selection.** Open it at
  a version bump, not during iteration.

## Handing off unfinished work

When you find a real problem you are not fixing in this session, write it into
**[ISSUES.md](ISSUES.md)** — do not leave it in the conversation, where the next
agent will never see it. Give each entry enough that someone can act on it cold:
the symptom, a command that reproduces it, what to do, and how to know it is
fixed. Delete the entry when it is fixed; `VERSION_HISTORY.md` is where the fix
gets recorded.

Read ISSUES.md before starting work — what looks like a fresh bug is often
already written up there.

## Important Notes

- **Do not overfit**: ~1,500 training rows. Ridge before XGBoost, XGBoost before anything larger, each step justified by a paired CV improvement.
- **Cap % normalization**: always convert raw dollars to cap %, never mix eras without normalization.
- **CBA regime awareness**: the 2023 CBA changed contract structure significantly. `cba_era` is the minimum encoding; `max_eligible_pct` encodes the actual ceiling per player.
- **Avoid double counting**: if availability is already reflected in minutes, don't apply a separate availability discount on top.
- **Scraping courtesy**: cache everything, rate-limit, don't hammer servers. Basketball Reference rate-limits aggressively — a full 30-team refresh takes ~25 minutes with backoff and one team will typically 403 and need a retry.
- **A failed fetch must degrade to stale data, never to missing data.** Merges that replace a whole season wholesale will silently delete a team whose page failed.
