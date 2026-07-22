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

## 2. `prev_cap_pct` is stale for 68 training rows

**Severity**: medium — one feature of fourteen, 2.2% of rows, all in training.

`prev_cap_pct` is "year-1 cap_pct of the player's previous contract", derived in
`scripts/phase3.py`. It was computed from `cap_pct` values that used the wrong
2025 and 2026 caps, so any row whose previous contract started in those seasons
carries a figure inflated by ~9.5% (2025) or deflated by ~7.8% (2026).

`load_training_data()` in `src/model/train.py` now recomputes `cap_pct` from
`config.CAP_BY_SEASON` on every load, but it does **not** recompute
`prev_cap_pct` — that derivation lives in phase3 and is not a simple shift.

**Reproduce**:

```bash
python -c "
import pandas as pd
t = pd.read_csv('data/processed/training_data_v2.csv')
y1 = t[t.year_in_contract == 1]
bad = set(zip(y1[y1.season.isin([2025, 2026])].player_name_norm,
              y1[y1.season.isin([2025, 2026])].cap_pct.round(6)))
n = sum((r.player_name_norm, round(r.prev_cap_pct, 6)) in bad
        for r in t.itertuples() if pd.notna(r.prev_cap_pct))
print(f'{n} contaminated rows')
"
```

**Fix**: re-run `python scripts/phase3.py` to regenerate `prev_cap_pct` against
the corrected caps, then retrain. Check first what else phase3 rewrites — it
touches several files in `data/processed/`.

**Verify**: the count reaches 0, and CV R² stays in the same neighbourhood (a
large move means phase3 changed more than intended).

---

## 3. Thirteen year-1 rows are paid above their own `max_eligible_pct`

That is impossible for a genuinely fresh contract, so each one is a bug
somewhere. They turn out to have **three unrelated causes** — do not try to fix
them as one thing.

**Reproduce the whole set**:

```bash
python -c "
import pandas as pd
d = pd.read_csv('outputs/web/valuations_export.csv')
y1 = d[d.year_in_contract == 1]
over = y1[y1.actual_cap_pct > y1.max_eligible_pct + 1e-9]
print(over[['player_name','season','actual_cap_pct','max_eligible_pct']].to_string(index=False))
"
```

### 3a. The award trigger is off by one season

**Severity**: medium — it understates the ceiling for genuine supermax players,
which makes the Grabit censoring gate miss them.

`_compute_max_eligible()` in `src/model/train.py` decides Rose Rule and Supermax
eligibility by asking whether a player was All-NBA/MVP/DPOY **in the season the
contract starts**. Those awards are earned in the season *before* a contract is
signed — that is what a team is paying for. Checking season `s` instead of `s-1`
means a player who earned his supermax in year `s-1` is scored against the
ordinary ceiling.

Confirmed cases: Ben Simmons 2021 should be 0.30 (Rose Rule), computed 0.25.
Jaylen Brown 2024 should be 0.35 (Supermax), computed 0.30 — he made All-NBA
2nd Team in 2023 and signed that summer.

**Fix**: in `_compute_max_eligible`, evaluate the trigger against `s-1`:

```python
trig = (p, s - 1) in elite_set or _elite_count(p, [s - 3, s - 2, s - 1]) >= 2
```

Retrain afterwards — this changes `max_eligible_pct`, which changes both the
Stage-2 clip and which rows the censoring gate admits.

**Verify**: Simmons 2021 reaches 0.30 and Brown 2024 reaches 0.35; five of the
thirteen rows above drop out (three of those five are the float ties in 3c).

### 3b. Extensions signed years before they begin

**Severity**: low — 2 rows, and a proper fix needs data the repo does not have.

John Wall 2019 and Karl-Anthony Towns 2024 are both paid 35% against a computed
ceiling of 0.30. Both signed supermax extensions two years before the deal
began, so their eligibility was judged on a season outside any window anchored
to the start season. Fixing 3a does not reach them.

**Fix**: needs a signing date per contract, which is not currently scraped.
Spotrac exposes it. Until then, leave these — do not widen the award window to
paper over it, as that would hand supermax ceilings to players who never
qualified.

### 3c. Escalator years still labelled year-1 before 2022

**Severity**: medium — corrupts the definition of a "fresh signing".

Six rows are later years of older deals wearing a year-1 label: Stephen Curry
2019 at 36.9% of the cap, John Wall 2020 at 37.8%, Chris Paul and Russell
Westbrook 2019 at 35.3%, Andrew Wiggins and CJ McCollum 2019 marginally over
25%. No max tier produces those figures on a fresh contract.

The same root cause explains why `scripts/check_caps.py` reports a low hit rate
for 2019 and 2020 (10/45 and 3/10) although both caps are correct: most of those
"year-1" max contracts are not year-1, so they land nowhere near a max tier.

Three further rows — Bam Adebayo and Jayson Tatum 2021 at exactly 0.250000,
Giannis Antetokounmpo 2021 at 0.350001 — are float noise against the ceiling,
not real violations. A 1e-6 tolerance removes them.

**Fix**: `data/processed/contract_structure_v2.csv` is the source of
`year_in_contract`. Either improve the match for 2019–2021, or add a validation
pass that demotes any year-1 row whose salary exceeds its max-eligible ceiling
by more than rounding — that condition is proof of mislabelling, not a judgement
call. Do 3a first, or the validation will demote rows that 3a should have fixed.

**Verify**: the reproduce query returns nothing, and `check_caps.py` stops
reporting a low hit rate for 2019 and 2020.

---

## 4. Site and docs quote CV R² from different runs

**Severity**: low — 0.0002 apart, but they are the same headline number.

`PROJECT_BRIEF.md` gives Grabit CV R² as 0.7581 (10-seed averaged);
`outputs/models/grabit_results.json`, which the site reads through
`export_web.py`, holds 0.7583 from a single seed. Both are defensible, but a
reader comparing the page against the brief sees two answers.

**Fix**: decide which is canonical. If it is the 10-seed average, have
`train.py` write that into `grabit_results.json` so the export inherits it
automatically.

---

## 5. Signing mechanism covers only 26% of rows

**Severity**: low — a known data limitation, recorded so it is not rediscovered.

`data/processed/spotrac_signing_types.csv` matches 818 of 3,113 player-seasons.
The site surfaces this in its limitations panel.

Keep the field out of `FEATURE_COLS` if you extend the scrape — `METHODOLOGY.md`
documents that modelling it *lowers* CV R² by 0.0073, and its value is as an
out-of-sample diagnostic.

---

## 6. Team and position missing for 5.4% of rows

**Severity**: low — cosmetic on the site, unused by the model.

167 of 3,113 rows have no `team_abbreviation` or `position` after the join
against `data/processed/training_data.csv`. The valuation board renders these
as "—". Neither field is a model feature, so this affects presentation and the
team filter only.
