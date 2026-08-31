# Handoff — the Spotrac merge layer's `status_active` branch

Written 2026-08-30 to be read cold. Read `CLAUDE.md` and `ISSUES.md` first.
**Nothing is adopted.** `training_data_v2.csv` and `salaries.csv` are still the
BBRef values, no model was trained, no evaluation ran.

---

## 1. What exists

The salary source is migrating from Basketball Reference to Spotrac because
BBRef's future-season columns are unreliable in three ways (ISSUES #50): stale
cap projections, declined options never removed, and dead money booked as
salary with no marker.

Pipeline, all offline off the HTML cache:

| file | role |
|---|---|
| `scripts/build_spotrac_urls.py` | builds `spotrac_player_urls.csv` (884 rows) |
| `scripts/resolve_spotrac_slugs.py` | resolves URLs the FA-page harvest misses |
| `scripts/build_spotrac_salaries.py` | parses pages → `spotrac_salaries.csv` |
| `scripts/build_merged_salaries.py` | picks ONE number per row → `merged_salaries.csv` |

`src/features/build_dataset.py` already consumes `merged_salaries.csv` and
drops `retained_only` rows. Coverage is 99.7% of training players.

## 2. Where it stands — evaluation frame, 880 rows

| branch | rows | exact vs BBRef | rate |
|---|---:|---:|---:|
| `spotrac_cap_hit` | 269 | 235 | 87.4% |
| **`status_active`** | **356** | **255** | **71.6%** |
| `vet_min` | 118 | 101 | 85.6% |
| `single` | 89 | 78 | 87.6% |
| `retained_only` | 22 | — | rows are DELETED |
| `split` | 12 | 10 | 83.3% |
| `waived` | 5 | 5 | 100% |
| `fallback_bbref` | 9 | 9 | 100% |

About 167 of 880 rows (19%) change. **`status_active` is the target** — it is
the largest computed branch and by far the weakest.

"Exact vs BBRef" is a diagnostic, not the goal. BBRef is wrong on some of these
rows by construction; do not optimise toward 100%.

## 3. The one fix with clear evidence

**Vet-minimum rows are landing in `status_active` instead of `vet_min`.**

```
Kyle Korver   2019   Active 2,404,456   BBRef 1,620,564
Dwight Howard 2019   Active 2,404,456   BBRef 1,620,564
Rajon Rondo   2021   Active 1,487,859   BBRef 1,669,178
```

Each BBRef value is the veteran-minimum CAP CHARGE; each Active amount is the
PAID minimum. They should be converted, not taken at face value. The detector
misses them because it tests the raw amount while 2019 Active figures are
COVID-reduced: `2,404,456 / 0.9375 = 2,564,753`, which is on the minimum scale.

**Do**: after `_status_base_candidate` returns a `status_active` value, run the
minimum-scale test on the COVID-adjusted amount before accepting the branch, and
route to `vet_min` (`_min_cap_charge`) on a hit. Reuse `_is_min_amount` /
`_min_cap_charge`; do not write a second minimum table.

Keep `MIN_MATCH_TOL_USD = 5_000`. At $25K the band swallows coincidences — one
team's share of Tyus Jones's 2025 season sits $13K from the 5-year minimum.

## 4. The hard part — do not burn time here without new evidence

Within `status_active`, a `Active;Retained` row can need either reading and the
labels do not separate them:

```
Gallinari 2022     Active 6,479,000 | Retained 13,000,000   BBRef 6,479,000  -> Active only
Gary Payton II 22  Active 2,814,368 | Retained  5,485,632   BBRef 8,300,000  -> sum
```

Signals already tried and measured, so do not re-derive them:

- **Team-badge count** (`n_teams`). Best any first-vs-sum rule can do is 68%.
- **`spotrac_transactions.csv`** as the trade flag. Incomplete — 878 traded
  player-seasons against 1,142 parseable off the pages.
- **Page transaction log** (`parse_transactions` in `build_spotrac_salaries.py`,
  already written and wired). Worth +138 rows against −10 on `status_active`
  and it IS the current rule, but it misfires in both directions on the two
  rows above.
- **`Buyout` exclusion inside a traded sum.** Correct in principle, worth
  almost nothing in practice — the problem rows carry no `Buyout` label.
- **Annuity detection** (an amount repeating an earlier season's amount).
  Tested; ~no gain.
- **Status `Active`-only, applied globally.** 14%. It halves every genuine
  traded season.

If you attack this, the missing signal is probably *whether he was on the
roster* that season, not anything in the money or the labels. Games played is
on the page and has not been tried.

## 5. Facts that cost a lot to establish — do not re-derive

- **Season convention.** Spotrac's per-year tables label by START year, same as
  the repo. The "Career Earnings thru YYYY" AGGREGATE uses the ENDING year.
  Two fields on one page, two conventions.
- **`Retained` does NOT mean dead money.** It means "paid by a team that is not
  his current team", which covers both a stretch annuity AND the pre-trade half
  of a traded season. This is the root of §4.
- **`Reserve` is a live contract**, not dead money. Excluding it recovered
  Spencer Dinwiddie 2019 ($10.6M) and DeAndre Jordan 2019 ($9.9M).
- **`NBA Cup` is tournament prize money**, never salary. Always drop it, and
  drop it before judging whether a row is "pure" dead money.
- **Split-cell rules are opposite by column.** In the career `Base` cell the
  FIRST amount is the contract's own; in the `Cash Cumulative` cell the LAST is
  the running total. The page's own `Total` row is the authority.
- **2019 is COVID-reduced** — Spotrac's career `Base` is 15/16 of contracted.
  Anchor: Al Horford's contracted $28,000,000 prints as $26,250,000. The
  divisor over-corrects about 2 frame rows (players who missed the season, e.g.
  Kevin Durant); no page signal separates them.
- **One slug carries several names.** Spotrac files a player under his legal
  name while box scores use the short one (`mohamed bamba` / `mo bamba`). A
  `dict(zip(slug, name))` lets the alias win and silently loses every season of
  6 players. Emit a row per name.
- **`page_defect` must run before parsing any cached page.** Two search-redirect
  URLs resolved to the NFL players of the same name and put 17 rows of NFL
  salary in the dataset. The page title carries the RIGHT name; only the
  canonical URL's sport betrays it.

## 6. Constraints

- Do NOT modify `training_data_v2.csv`, `salaries.csv`, `config.py`, or
  anything under `src/`. This work produces a candidate table and its audit.
- Do NOT run `rebuild_training_data.py`, model training, or any evaluation
  suite. Adoption is the user's call.
- Do NOT make network requests. Everything needed is cached.
- `scripts/build_merged_salaries.py` prints its own audit; run it with no flags
  for a dry run and `--write` to persist.
- When you change a branch, report the frame table from §2 before and after,
  and the count of rows fixed versus broken. A branch rate that moves less than
  a percentage point is not a result.

## 7. Also open, lower priority

- `resolve_spotrac_slugs.py` has no rate limit. An unthrottled run 403'd 34 of
  ~120 requests; all 34 succeeded later at 3.5s spacing. See ISSUES #53.
- 18 `redirect/player/<id>?ref=search` URLs remain in the URL table. They are
  the only ones that can land on the wrong sport.
  `build_spotrac_urls.py --repair-redirects` replaces 9.
- ISSUES #41 is effectively fixed (15 of its 16 players now have data) and
  should be deleted; #28, #36 and #43 need updating.
