# RESULT — five-arm feature batch (trend / timing / Arm B / est-value / rookie awards)

Pin: `master` @ v7.10x (944-row frame, repaired `prev_cap_pct`).
Branch: `worker/feature-batch`. Harness: one paired run, 10 seeds × 5
GroupKFold folds, same-run incumbent, per-fold deltas on the selection pool
(`fold_r2_selection`), the canonical `evaluate_suite.paired_delta` unit.

## 1. What changed and why

Five candidate feature arms were built as ship forms and scored against a
single same-run incumbent (the current 14-feature Grabit champion). **Only arm
(c) — the Arm B `prev_cap_pct` semantics swap — moves the model, and it lands
exactly on the gate boundary** (ΔSel +0.00178, per-fold t 1.96). Every other arm
is flat or harmful on the selection pool: the trend features (a), the
prior-year estimated value (d1), and the rookie-award ordinal (e) each add
essentially nothing on top of the existing features — DARKO already encodes
trajectory and the existing awards channel already carries the pedigree signal,
exactly as the brief anticipated. The stats-as-of-signing replacement (b) is
actively destructive, and the in-slot est-value replacement (d2) degrades the
breakout kill-zone the brief flagged.

Two ship-form traps bit during the work and are worth carrying forward. First,
a naive in-harness Arm C left 89 true-first-contract rows at native NaN because
the *evaluation* frame has no rookie-scale rows to calibrate the fill map; the
shipped builder fills them (≈0.0148) identically to Arm A, so the honest
semantics-swap delta is +0.00178, not the +0.00479 the buggy column produced.
Second, the prior prev-cap RESULT's "+0.0020 (t=7.0)" is reproduced here in
**magnitude** but its t came from pairing on **seeds** (seed sd ≈ 0.001), not
folds — the canonical per-fold pairing on the same matrices gives t = 1.96 (see
§4). Every failing arm's columns are still delivered on the branch for the route
classifier.

## 2. Numbers

Incumbent (same-run Grabit champion, 944 rows, 14 features):
**A1 0.7849 · A2 0.8338 · B1 0.8276 · pooled MAE $3.158M.**

| arm | what | A1 | A2 | B1 | MAE$M | ΔSel | SE | t(fold) |
|-----|------|----|----|----|-------|------|----|---------|
| **c** | prev_cap_pct = prev-season pay, uniform | 0.7856 | 0.8349 | 0.8298 | 3.135 | **+0.00178** | 0.00091 | **+1.96** |
| a | +9 trend cols (native NaN) | 0.7872 | 0.8327 | 0.8229 | 3.170 | −0.00015 | 0.00312 | −0.05 |
| a_cov | trend + coverage indicator | 0.7872 | 0.8332 | 0.8232 | 3.170 | +0.00010 | 0.00295 | +0.03 |
| b_replace | prod features ← as-of-signing | 0.7567 | 0.7990 | 0.8002 | 3.396 | **−0.03218** | 0.00731 | −4.40 |
| b_delta | +6 signdelta cols + is_priced_early | 0.7969 | 0.8420 | 0.8364 | 3.042 | +0.01083 | 0.00863 | +1.25 |
| d1 | +est_value_prev (native NaN) | 0.7871 | 0.8334 | 0.8239 | 3.167 | +0.00008 | 0.00244 | +0.03 |
| d1_cov | est_value_prev + coverage | 0.7871 | 0.8332 | 0.8233 | 3.164 | +0.00007 | 0.00227 | +0.03 |
| d2 | in-slot prev_cap_pct ← est (rookie-exit) | 0.7813 | 0.8305 | 0.8254 | 3.208 | −0.00476 | 0.00256 | −1.86 |
| e_new | +rookie_award_tier ordinal | 0.7850 | 0.8339 | 0.8261 | 3.168 | +0.00017 | 0.00118 | +0.15 |
| e_cleanonly | award_score_cum, cleaned names (control) | 0.7807 | 0.8285 | 0.8238 | 3.181 | −0.00422 | 0.00252 | −1.67 |
| e_reweight | award_score_cum, cleaned + rookie-up | 0.7810 | 0.8272 | 0.8243 | 3.174 | −0.00345 | 0.00258 | −1.34 |

Per-fold ΔSel (the decision unit, 5 values):

```
c           [ 0.00205, -0.00011, -0.00031,  0.00275,  0.00451]   mean +0.00178
a           [ 0.00276,  0.00046,  0.00335,  0.00499, -0.01230]   mean -0.00015
b_replace   [-0.03973, -0.01269, -0.04146, -0.01703, -0.05000]   mean -0.03218
b_delta     [ 0.01166, -0.00491,  0.01778,  0.03891, -0.00929]   mean +0.01083
d1          [ 0.00230,  0.00486,  0.00482, -0.00614, -0.00543]   mean +0.00008
d2          [-0.00520,  0.00037, -0.00025, -0.00476, -0.01394]   mean -0.00476
e_new       [ 0.00044, -0.00312, -0.00161,  0.00366,  0.00149]   mean +0.00017
e_cleanonly [-0.00261, -0.00956, -0.00987,  0.00361, -0.00266]   mean -0.00422
```

Control arms run: **a_cov** and **d1_cov** (coverage-indicator control, gate 2 —
missingness aligns with injury/experience); **e_cleanonly** (name-fix control,
isolating the reweight from the name-cleaning); **b_delta**'s `is_priced_early`
indicator is the coverage arm folded into the delta variant.

## 3. Gate verdicts

Gates per arm: (1) ΔSel ≥ +0.002 with t > 2; (2) coverage/season control where
alignment applies; (3) B1 forward drop ≤ 0.003 moving *with* the pooled gain;
(4) no fixed-segment |bias| growth > $0.3M.

| arm | G1 ΔSel/t | G2 control | G3 B1 | G4 C2 | verdict |
|-----|-----------|-----------|-------|-------|---------|
| **c** | **borderline** +0.00178 / t 1.96 | n/a | +0.0022 ✓ | max +$0.031M ✓ | **at the boundary** |
| a | ✗ −0.00015 / −0.05 | a_cov no better | −0.0047 ✗ | breakout +$0.303M ✗ | FAIL |
| a_cov | ✗ +0.00010 / 0.03 | ≈ a | −0.0044 ✗ | breakout +$0.308M ✗ | FAIL |
| b_replace | ✗ −0.03218 / −4.40 | — | −0.0274 ✗ | breakout +$2.19M ✗ | FAIL (harmful) |
| b_delta | ✗ +0.01083 / **1.25** | is_priced_early in-arm | +0.0088 | Cap Space +$0.55M, S&T +$0.75M ✗ | FAIL (t<2, C2) |
| d1 | ✗ +0.00008 / 0.03 | d1_cov no better | −0.0037 ✗ | S&T +$0.27M ✓ | FAIL |
| d1_cov | ✗ +0.00007 / 0.03 | ≈ d1 | −0.0043 ✗ | S&T +$0.25M ✓ | FAIL |
| d2 | ✗ −0.00476 / −1.86 | — | −0.0023 | breakout +$0.22M | FAIL (harmful) |
| e_new | ✗ +0.00017 / 0.15 | — | −0.0015 | season2019 +$0.03M ✓ | FAIL |
| e_cleanonly | ✗ −0.00422 / −1.67 | — | −0.0038 ✗ | Non-Bird +$0.15M ✓ | FAIL |
| e_reweight | ✗ −0.00345 / −1.34 | e_cleanonly control | −0.0033 | Non-Bird +$0.18M ✓ | FAIL |

**Adoption recommendation.** Adopt **arm (c)** as the lead (and only) candidate,
as a *semantics correction* rather than a gate-passing feature win — the
architect makes the boundary call. Its case: (i) magnitude +0.00178 reproduces
the pre-registered +0.0020 from the prev-cap RESULT; (ii) A1, A2, B1 and MAE all
improve in the same direction; (iii) C2 is pristine (max fixed-segment |bias|
growth +$0.031M, two orders under the gate); (iv) the watchlist segments
(breakout, Early Bird, prime, 2019) are all flat, ±$0.02M. The single reservation
is the canonical per-fold t of 1.96, a hair under 2 — see §4. No re-pairing chain
is needed because nothing else clears; (c) stands alone.

## 4. Anomalies (flagged, not victories)

- **Arm (c) t reconciliation.** Same ΔSel magnitude, three pairing conventions
  on the identical `fold_r2_selection` matrices:
  per-fold (n=5, canonical) t **1.96**; per-seed (n=10) t **6.07**; all-cell
  (n=50) t 5.37. The prev-cap RESULT's "t=7.0" is the per-seed figure — seed sd
  ≈0.001 treats RNG as the only noise and inflates t ~3×. The worker brief's
  mandated unit is the fold, so 1.96 is the honest number; the magnitude, not
  the seed-paired t, is what actually reproduces.
- **Arm (c) buggy vs ship form.** A first pass scored (c) at +0.00479 (t 2.47).
  That was an artifact: the eval frame carries no rookie-scale rows, so the
  hand-rolled fill map silently left 89 true-first-contract rows at native NaN
  where Arm A fills them identically — the delta partly measured NaN-vs-fill.
  The ship form (fill via `build_contract_features` on the full frame) is
  +0.00178. Byte-for-byte parity of the delivered column vs the shipped builder
  is exact (max abs diff 1e-16, 0/944 mismatches). Arm C differs from the baked
  Arm A on 232/944 rows.
- **b_delta ΔSel +0.01083 is above the +0.01 re-verify line but is not real.**
  Mechanism: the 133 priced-early extension rows are dominated by Bird-Rights
  re-signings, and the delta columns let the model discount their stale
  production — the Bird Rights segment bias does improve, −$2.51M → −$1.84M (the
  brief's stated target). But the per-fold pattern is unstable (one fold +0.039
  carries it, two are negative), the selection t is only 1.25, and C2 breaks in
  two segments (Cap Space +$0.55M, Sign & Trade +$0.75M) and drifts 2019
  (+$0.26M). A targeted segment win that does not survive the pooled gate or C2.
- **b_replace wrecks the breakout class.** Wholesale replacement moves breakout
  (age≤24 & ≥$15M) bias −$2.87M → −$5.06M: young players on rookie extensions
  were *better* at their current season than at signing−1, so substituting the
  earlier, weaker production makes the model underpay them further. The
  MPJ-style "priced pre-injury" case is real but rare; the replacement's average
  effect on the extension population is strongly negative.
- **Award name-join bug (new ISSUES entry).** 90 of 943 rows in
  `awards_full.csv` carry footnote junk in `player_name_norm` (`^ § † digits`,
  the replacement char), which `build_external_features.norm` does not strip, so
  `award_score_cum`/`all_nba_cum` silently drop award seasons for the biggest
  stars — every one of the 13 ROY winners, plus MVP/All-NBA rows for Durant,
  Curry, Giannis, Jokić, Harden, Westbrook. **Counter-intuitively, fixing it
  slightly *hurts* CV** (e_cleanonly ΔSel −0.00422): the recovered superstar
  award mass is redundant with `darko`/`prev_cap_pct` and the players are
  ceiling-pinned, so it adds variance without lift. Real data-quality defect,
  low model priority — documented, not fixed here (see ISSUES).

## 5. Ship-form statements

- **(c)** `scripts/phase3.build_contract_features(prev_mode="prev-season")` is
  the ship form; the delivered `prev_cap_pct_prevseason` reproduces it
  byte-for-byte (verified). `prev_mode` defaults to `"year1-loop"` (Arm A,
  unchanged); nothing ships until the architect flips the default and rebuilds
  `training_data_v2.csv`.
- **(a)** native-NaN throughout (XGBoost default-direction handling); no median
  fill. `trend_has_prev` is the coverage arm.
- **(b)** delta form keeps the current production intact and adds
  `{feat}_signdelta` = current − as-of (0 where inapplicable/missing) plus
  `is_priced_early`; the replace form is provided as `*_asof` columns but is
  destructive and not a shippable feature.
- **(d)** `est_value_prev` native-NaN; `$/win` is a per-season constant derived
  from the cap (`N_TEAMS·cap / 1230`) and cancels against the cap, so the column
  is an era-neutral cap share — no fitting. d2 clips into
  `[floor_pct, max_eligible_pct]` on rookie-exit rows only.
- **(e)** ordinal only (ROY=3 / AR1=2 / AR2=1 / 0), names cleaned so ROY winners
  match; no empirical dollar mapping.

## 6. Delivered columns (for the route classifier)

`data/processed/feature_batch_columns.csv` — 944 rows keyed by
`(player_name_norm, season)`, regenerated by `python scripts/feature_batch.py`:

- **(a)** `darko_dpm_z_d1/_slope3/_peakd`, `lebron_z_…`, `rapm_z_…` (9),
  `trend_has_prev`
- **(b)** `darko_dpm_z_signdelta`, `lebron_z_signdelta`, `rapm_z_signdelta`,
  `mpg_signdelta`, `usage_pct_signdelta`, `ast_pct_signdelta`, `is_priced_early`
- **(c)** `prev_cap_pct_prevseason`
- **(d)** `est_value_prev`, `est_has_prev`, `prev_cap_pct_d2`
- **(e)** `rookie_award_tier`, `award_score_cum_clean`, `award_score_cum_reweight`

The replace-form `*_asof` columns are available from
`scripts.feature_batch.asof_replace` if the classifier wants them.

## 7. Files touched

- `scripts/phase3.py` — `build_contract_features` gains a `prev_mode` argument;
  default `"year1-loop"` is byte-identical to before, `"prev-season"` is Arm C.
- `scripts/feature_batch.py` — new: ship-form builders for all five arms + a
  `main()` that writes the column manifest.
- `data/processed/feature_batch_columns.csv` — new: the delivered columns.
- `ISSUES.md` — new entry on the award name-join defect.
- `docs/briefs/2026-07-25-feature-batch.RESULT.md` — this file.

Scratch harness (not committed): the paired runner and per-arm builders live in
the session scratchpad; `scripts/feature_batch.py` is the committed, reproducible
form of every builder.

## 8. Proposed commit message (architect edits and lands)

```
Feature batch: 5 arms scored; only Arm B prev_cap_pct semantics survives (boundary)

Paired harness (10s x 5f, per-fold selection pool) over five candidate arms:
- (c) prev_cap_pct = prev-season pay, uniform: ΔSel +0.00178, per-fold t 1.96
  (per-seed t 6.07 — the prev-cap RESULT's "t=7" was seed-paired). A1/A2/B1/MAE
  all improve, C2 max +$0.031M. build_contract_features gains prev_mode; ship
  form verified byte-for-byte. At the gate boundary — semantics fix, architect's
  call.
- (a) trend, (d1) est-value, (e) rookie-award tier: flat on selection (DARKO /
  existing awards already carry the signal).
- (b) stats-as-of-signing: replace form harmful (breakout −$2.9M→−$5.1M), delta
  form +0.011 but t 1.25 and C2 breaks (Cap Space +$0.55M).
- (d2) in-slot est on rookie-exit: harmful, breakout kill-zone as warned.
All arms' columns delivered in data/processed/feature_batch_columns.csv for the
route classifier. New ISSUES entry: awards_full name-join drops footnote-marked
stars from award_score_cum (real defect; fixing it slightly hurts CV).
```
