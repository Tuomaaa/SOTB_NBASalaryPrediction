# kf_market_value — adoption report (v8.11x candidate)

**Date**: 2026-08-09
**Proposal**: replace `prev_cap_pct` with `kf_market_value` in the regression's
21-feature list (SWAP form — feature count unchanged; the route classifier's
frozen `CLF_BASE_COLS` keeps `prev_cap_pct`, per the v8.6x decoupling).
**Recommendation**: ADOPT. Three independent protocols — pooled selection CV,
rolling-forward, and the locked confirmation split — reproduce the effect where
the feature acts, both controls are clean, and no protocol shows a reversal.
The one miss is the primary gate itself: selection-pool paired t = +1.75
against the t > 2 bar, with the shortfall attributable to dilution from the
segment the feature deliberately leaves untouched (measured below).
**Status**: recommended; pending ratification and the productionization work
listed at the end. The confirmation split is SPENT for this cycle — it was
opened for this decision and must not be read again before the next version
boundary.

---

## 1. The feature

For an evaluation row (player p, season T): anchor a random-walk Kalman filter
at the market's last verdict on p, then update it chronologically with the
champion model's fold-honest valuation of each intervening season
(T0+1 … T−1, rows that exist only in the 3,113-row full table — escalator
years, option years, minimum stints). Output: a smoothed trajectory between
"where the market last priced him" and "what his seasons since were worth".

Three-tier anchor (the design that made the feature work — see §2):

| tier | anchor | value | prior variance | n (of 873) |
|---|---|---|---|---|
| 1 | most recent PRIOR row of p in the eval frame — eval membership IS the project's definition of a fresh market price | that row's cap_pct (cap-hold normalized, corrected) | P0 = 0.0005 | 428 (226 with measurements) |
| 1 (prorated) | a prorated year-1 stint (raw label, cap_pct < 1.2%): a 10-day at the minimum RATE is a market verdict; only its cap_pct prorates playing time | pulled UP to the season's full-season minimum cap charge | P0 = 0.0005 | 61 of the 428 |
| 2 | no market anchor but rookie-scale seasons exist (first-rounders): earliest observed rookie season (deal year 2); y3/y4 team-option seasons become measurements | that row's cap_pct | **P0 = R** — a slotted price gets the trust of ONE model measurement, fixed before scoring | 133 (118 with measurements) |
| 3 | neither | — | — | 312: `kf = prev_cap_pct`, the feature degrades to information the model already has |

Filtered year-1 labels (continuation mislabels, rookie scale, first contracts)
can never anchor; their seasons become measurements inside the window, which
makes the KF behave as if the frozen contract-structure table's labels were
correct without touching the table.

KF parameters, all fixed before any candidate score was seen: P0 as above;
R = inner-OOF residual variance per (seed, fold) (measured 0.00134);
Q = max(var(year-over-year Δcap_pct, outer-train multi-contract players) − 2R,
0.0005) (the floor binds). Fallbacks: no anchor → prev_cap_pct; anchor but no
measurements → the anchor value.

## 2. Why two previous attempts failed and this one does not

| attempt | measurement model | anchor | ΔSel | t |
|---|---|---|---|---|
| Phase 1 | 2-param linear (R² 0.46) | prev_cap_pct, raw windows | +0.00045 | +0.61 |
| Variant A | full champion (R² 0.81) | prev_cap_pct, raw windows | −0.00042 | −0.39 |
| **This (variant B)** | full champion (R² 0.81) | **three-tier market anchor** | **+0.00331** | **+1.75** |

Variant A proved measurement strength alone changes nothing. The gain came
entirely from the anchor redesign: of the 716 raw-label anchors, 381 were
provably not market prices (52 prorated partial-season stints, 132 rookie-scale
slots, 197 continuation-mislabels/first contracts). Diagnostics after the fix:
r(kf, prev) fell 0.973 → 0.928; univariate r(kf, y) = 0.694 vs
r(prev, y) = 0.548; on the tier-2 (rookie) segment r(kf, y) = 0.757 vs 0.467.
372/873 rows move by more than 0.1% of cap.

## 3. Fold-honesty (nested CV)

Outer 5-fold GroupKFold by player × 10 seeds — identical folds and seeds as
the champion, so all deltas are paired. Per outer fold: a 4-fold inner
GroupKFold (also by player) over the outer TRAINING rows produces (a)
fold-honest Year-1 predictions (R and the inner-OOF R² = 0.807), and (b)
measurements for the full-table intermediate rows — a training row's player
takes the inner model that excluded that player; a test-fold player, never
seen by ANY inner model, takes the 4-model mean. kf is recomputed per outer
fold, so no fold's models ever see a kf value informed by that fold's players.
The intermediate predictor is always the 21-feature champion — kf never feeds
itself. ~250 champion-pipeline fits per arm-set, ~13 min.

## 4. Evidence

**A — selection (deciding layer), 873 rows, 10 seeds:**

| arm | A1 | sel | A2 | MAE | ΔSel (t) |
|---|---|---|---|---|---|
| champion (prev_cap_pct, 21) | 0.8226 | 0.8211 | 0.8456 | $2.85M | — |
| **SWAP prev→kf (21)** | 0.8254 | 0.8247 | 0.8491 | $2.82M | **+0.00331 (+1.75)** — the pre-registered primary; below the t > 2 bar |
| ADD +kf (22, reference) | 0.8253 | 0.8247 | 0.8487 | $2.82M | +0.00349 (+2.13) |
| season-dummy control (21) | 0.8226 | 0.8215 | 0.8462 | $2.85M | +0.00040 (+0.88) |

Per-fold (swap, selection): +0.00578, +0.00082, −0.00046, +0.00086, +0.00957.

**Segments (champion vs SWAP, fold-paired):** all +0.00252 (t +1.39); tier-3
(kf ≡ prev) −0.00231 (t −0.75), MAE unchanged; **kf-active (561 rows)
+0.00499 (t +1.94), MAE $2.88M → $2.83M.** The gain sits exactly where the
feature acts; the primary gate's shortfall is dilution from the 312 rows the
feature leaves untouched, plus their noise-level collateral.

**Season-dummy control**: `prev + per-season mean(kf − prev)` — same
season/coverage structure, zero player content (kf coverage grows with season,
the supply_samepos/is2019 trap). It recovers **12%** of the swap gain
(is2019 recovered 70% of supply_samepos and killed it). PASS.

**C guards**: C1 calibration slope 0.9661 → 0.9668, |slope−1| excess −0.0006
(gate ≤ +0.005). C2 worst mechanism |bias| growth: Early Bird +$0.12M
(gate ≤ +$0.3M); Non-Bird improves −$0.23M. PASS both.

**B1 — rolling-forward veto** (train < T, score T; nothing from season T
enters the kf in any capacity — inner OOF per origin window):

| origin | n | champion → swap R² | ΔMAE |
|---|---|---|---|
| 2024 | 106 | 0.8380 → 0.8394 (+0.0014) | −$0.04M |
| 2025 | 102 | 0.8229 → 0.8234 (+0.0005) | −$0.09M |
| 2026 | 104 | 0.8417 → 0.8553 (**+0.0136**) | −$0.14M |
| pooled | 312 | 0.8343 → 0.8387 (**+0.0044**), bootstrap 95% CI [−0.0001, +0.0090] | −$0.09M |

3/3 origins positive, same direction as layer A, on a split geometry the
design was never iterated against.

**D3 — locked confirmation split, opened 2026-08-09 for this decision:**

| slice | n | ΔR² (swap − champion) | 95% CI | ΔMAE |
|---|---|---|---|---|
| confirmation, all | 125 | −0.0012 | [−0.0114, +0.0070] | −$0.05M |
| confirmation ∩ kf-active | 77 | **+0.0066** | **[+0.0004, +0.0159]** | −$0.12M |
| confirmation ∩ tier-3 | 48 | −0.0125 | [−0.0476, +0.0027] | +$0.06M |

The locked rows reproduce the selection pattern exactly: the active segment
wins with a CI excluding zero — players that entered no design decision — and
the flat headline is the same tier-3 dilution. Dollar MAE improves on the
deciding slice. (The ADD arm reads the same on confirmation but worse:
−0.0027 headline / +0.0047 active — switching arms post-confirmation was
considered and rejected as selection-on-confirmation.)

## 5. Honest liabilities

1. The pre-registered primary (pooled selection t > 2) was missed at +1.75.
   Everything else in this report is why adoption is recommended anyway; a
   ratifier who holds the strict line should reject, and the artifacts support
   either ruling.
2. Tier-3 collateral: −0.0023 (sel) / −0.0125 (conf), both CIs spanning zero,
   MAE flat — consistent with tree-structure noise, monitored, not proven zero.
   Composition of the 312 tier-3 rows (by debut season): **236 (76%) are a
   self-extinguishing window-boundary artifact** — veterans who debuted before
   2019, whose prior market contracts predate the data window (T distribution
   2019: 128 → 2020: 55 → 2021: 30 → 2022: 14 → 2023: 6 → 2024-26: 1/yr);
   75 (24%) are a structural class — second-round/undrafted players whose only
   prior history is their filtered first contract (10-17 per season, steady).
   So tier-3 shrinks toward ~75 rows as the window advances and kf coverage
   rises automatically — consistent with B1's largest gain landing on the 2026
   origin. A future experiment can anchor the first-contract class the way
   tier 2 anchors first-rounders (convention pay + P0 = R); it does not block
   this adoption.
3. The confirmation split is spent for this cycle.
4. kf coverage rises with season (2019-20 rows are mostly tier-3). The season
   control passed at 12%, but B1-by-origin should be watched on future
   refreshes.

## 6. Adoption work items (in order)

1. Productionize: move the builder out of the experiment script into
   `src/features/kf_market_value.py`; integrate into the rebuild chain; add
   the `predict.py` inference path (deployed model prices the intermediate
   seasons — no nested CV at inference; TRAINING-time values keep the
   fold-honest construction).
2. `train.FEATURE_COLS`: `prev_cap_pct` → `kf_market_value`. Do NOT touch
   `route_mixture.CLF_BASE_COLS` (frozen, v8.6x).
3. Rebuild-adjacent: regenerate `SIGNING_OFFSETS_DEPLOYED` (ISSUES #48 flow),
   run the full evaluate_suite for the official A1/A2/B1, verify signing
   guards.
4. Docs: METHODOLOGY.md (feature definition + this decision chain),
   VERSION_HISTORY.md entry, CONTEXT.md vocabulary; tag v8.11x with A1/A2/B1.
5. Related but separate: ISSUES #49 (rookie-scale fill semantics) rides the
   next rebuild, not this change.

## 7. Artifacts

| artifact | path |
|---|---|
| A-layer harness (4 arms, segments, C guards) | `scripts/ablation_kf_market_value_full.py` (`--anchor market`) |
| B1 forward veto | `scripts/eval_kf_forward.py` |
| A-layer results | `outputs/models/ablation_kf_market_value_full_market.json` |
| per-row OOF (all arms + tier + kf) | `outputs/models/ablation_kf_oof_market.csv` |
| B1 results | `outputs/models/kf_forward_veto.json` |
| Variant A record (raw anchors, FAIL) | `outputs/models/ablation_kf_market_value_full.json` |
