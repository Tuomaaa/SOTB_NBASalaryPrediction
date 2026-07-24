# Task brief: five-arm feature batch (trend / timing / Arm B / est-value / rookie awards)

Read `docs/worker-brief.md` first — isolation section, own worktree,
`OMP_NUM_THREADS=6`.

**Pin**: current `master` (v7.10x line, 944-row frame, repaired
`prev_cap_pct`). Deliver a branch +
`docs/briefs/2026-07-25-feature-batch.RESULT.md`.

Five arms, one paired harness (10 seeds × 5 GroupKFold folds, same-run
incumbent, per-fold deltas). Every arm that fails its regression gate still
gets its columns delivered on the branch — the route classifier (parallel
brief `2026-07-25-route-mixture.md`) consumes them regardless.

## The arms

**(a) Trend features.** Per impact metric (darko/lebron/rapm z): one-year
delta (t − t−1), three-year slope, and peak-minus-current (career-best z
minus current). Ingredients are in the cached per-season tables — self-join,
no scraping. DARKO already encodes trajectory internally, so the bar is
beating current-value + age curve. Missing t−1 (rookies, missed seasons):
native-NaN, and run the coverage-indicator control — missingness aligns with
injury/experience.

**(b) Stats as of signing.** For `is_extension` rows (via
`contract_signing_dates.csv` spans), the market priced the SIGNING-DATE
production, not the latest season (MPJ 2022: priced pre-injury). Arm: replace
the impact/production features with their values as of the last full season
before the signing date (or add the delta as a feature — test both forms,
ship-form discipline). Target: the $15-30M band and the Bird bias (−$2.8M).

**(c) Arm B semantics swap.** `prev_cap_pct` = previous SEASON's actual pay,
uniformly (drops the year-1 loop). Already validated in-memory at +0.0020
over Arm A (t=7.0) in the prev-cap RESULT; your job is the SHIP FORM: code it
in `build_contract_features`, verify the built column reproduces your tested
column byte-for-byte, then confirm the paired delta.

**(d) Prior-year estimated value** (user-proposed). est = prior-year
darko × mpg → wins → × $/win, with $/win a per-season CONSTANT derived from
the cap (no fitting — a fitted rate would be a mini-model with leakage
questions). Two sub-arms: (d1) new column; (d2) rookie-exit rows only,
in-slot replacement of `prev_cap_pct`, expressed as cap share and clipped
into [floor_pct, max_eligible_pct]. Known limit: inherits DARKO's blindness
to young high-usage scorers (helps the MPJ class, not the Booker class).

**(e) Rookie-award tier** (user-proposed). Ordinal: ROY=3, All-Rookie
1st=2, 2nd=1, none=0 — `awards_full.csv` has full 2014-2026 coverage. The
voter channel that catches exactly what (d) misses (Booker: All-Rookie 1st,
zero All-Stars at signing). FIRST check how `award_score_cum` currently
weights All-Rookie/ROY — if the signal is already inside and merely drowned,
reweighting may beat a new column; report both. A fixed-dollar fill version
is NOT in scope (it is the slot-average family: target-derived, C2-fried).

## Judging

Full challenger gates per arm, independently: paired ΔSel ≥ +0.002 with
t > 2 on the selection pool; A2 same direction; C2 fixed-segment |bias|
growth ≤ $0.3M; B1 forward drop ≤ 0.003. Coverage-aligned arms (a, b) add
the coverage-indicator control. Watchlist segments reported for every arm:
breakout (age≤24 & ≥$15M, bias −3.16 at v7.10x), Early Bird (the
slot-average kill zone), prime 25-29, season 2019.

Adoption order if several pass: (c) first (semantics fix), then largest
ΔSel first, each increment re-paired against the running base.

## Traps

- SHIP-FORM everywhere: the tested column and the shipped builder must match
  byte-for-byte (the prev-cap RESULT's standard).
- (d2) changes a top feature's meaning on 144 rows — the Early Bird segment
  is where the last attempt died; watch it specifically.
- (e) as ordinal only; no empirical dollar mapping.
- Deltas above +0.01 are a re-verify flag, not a victory lap.
- Do not touch filters, censoring, or Stage 2 — features only.

## Deliverable

Branch + RESULT: per-arm paired tables with per-fold deltas, watchlist
segments, ship-form statements, adoption recommendation with ordering, and
the delivered-columns manifest for the classifier. Expected effort: 1-2 days.
