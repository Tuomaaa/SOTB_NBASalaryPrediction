"""Nested-CV ablation: kf_market_value with the FULL champion model as measurement.

The feature: for a player pricing at season T whose previous Year-1 contract
was at season T0, anchor a Kalman filter at prev_cap_pct (the market's last
actual price signal) and update it with the champion model's predictions on the
player's intermediate seasons (T0+1 .. T-1, the escalator years in df_full).
The KF output is a smoothed trajectory between "where the market last priced
him" and "where his stats say he should be now".

A Phase-1 test used a 2-parameter linear predictor (base_rating + age,
R2=0.46) as the measurement: the trajectory concept showed signal (R2
0.25 -> 0.41 on 386 multi-contract pairs) but the feature failed the gate
(dSel=+0.00045, t=+0.61) because the weak predictor systematically dragged
star anchors down on long contracts. This script swaps in the full 21-feature
Grabit champion (R2~0.82 OOF) as the measurement, which requires nested CV
for fold-honesty:

    outer fold f (5-fold GroupKFold by player, 10 seeds):
        inner 4-fold GroupKFold OOF over the outer TRAINING rows
            -> fold-honest Year-1 predictions (diagnostic R2, and R)
            -> per-inner-model predictions on the df_full intermediate rows
        KF per row, using ONLY inner-world measurements:
            train row (player p in inner fold j): measurements from inner
                model j, which excluded ALL of p's Year-1 rows
            test row (player p in fold f): p never entered ANY inner model,
                measurements are the mean of the four inner models
        fit the 22-feature candidate on the outer training rows, score fold f

    The kf column is recomputed per OUTER fold: fold f's models never see a
    kf value that was informed by fold f's players. (A single per-seed kf
    vector reusing each row's own-fold value would let fold g's TRAINING rows
    carry inner-OOF information from fold g's players — the channel this
    per-fold recomputation closes.)

The intermediate predictor is always the 21-feature champion — kf_market_value
never feeds itself (two-stage, not feedback). The route classifier's feature
list is FROZEN (v8.6x) and does not grow with the regression's, so the
candidate adds kf_market_value to the regression alone.

KF parameters (Phase-1 values, kept):
    P0 = 0.0005          anchor is ground truth, near-zero prior variance
    R  = var(inner OOF residuals), per (seed, fold)
    Q  = max(var(year-over-year dcap_pct, outer-train multi-contract
             players) - 2R, 0.0005)

Fallbacks: no anchor value -> NaN (median-imputed); no intermediate seasons
(or no prior Year-1 contract on record) -> kf_market_value = the anchor.

Anchor design (--anchor):
    market (default) — three-tier anchor:
        tier 1  the player's most recent PRIOR row in the evaluation frame
                itself: the last provable fresh market pricing. Filtered
                year-1 labels (prorated stints, continuation mislabels,
                rookie scale) cannot anchor — their seasons become
                measurements inside the window instead. Anchor value is read
                from the EVAL frame (cap-hold normalized, salary-corrected);
                prior variance P0 = 0.0005 (a market price is near truth).
        tier 2  no market anchor, but the player has rookie-scale seasons
                before T (first-rounders): anchor at the EARLIEST observed
                rookie season (deal year 2, treated as the deal's pricing
                point), every later pre-T season is a measurement — the
                y3/y4 team-option trail is where a declining trajectory
                shows. A slotted price is not a market judgment, so its
                prior variance is R: the anchor gets the trust of ONE model
                measurement, no more. Fixed before this arm was ever scored;
                never re-tune it on a result.
        tier 3  neither -> kf = prev_cap_pct: the feature degrades to
                information the model already has.
    prev — variant A, kept reproducible: anchor = prev_cap_pct (v7.11x
        semantics, last season's pay) with the window drawn from the RAW
        year_in_contract labels, dirty anchors included. Superseded by
        'market'; its 10-seed result: dSel -0.00042, t -0.39, FAIL.

Two candidate arms per run, off the same inner OOF and the same kf values:
    ADD    21 champion features + kf_market_value (22)
    SWAP   prev_cap_pct -> kf_market_value (21) — the sharper test, since kf
           is a refinement of prev_cap_pct, identical wherever no
           intermediate season exists

Gate: paired selection-pool delta, t > 2 (CLAUDE.md, layer A).

Run:  python scripts/ablation_kf_market_value_full.py [--seeds 10]
          [--anchor prev|year1]
      (--seeds 1 is a ~4-minute smoke pass; the gate needs the full 10)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import OUTPUTS_DIR
from src.model.evaluate_suite import (
    N_SPLITS, load_evaluation_frame, make_champion_fitter, paired_delta,
    _dollars, _fold_pass, _reduce_fold_pass,
)
from src.model.route_mixture import (
    attach_clf_features, train_route_classifier, route_proba, MAX_IDX,
)
from src.model.stages import compose
from src.model.train import (
    FEATURE_COLS, TARGET, load_training_data, _compute_max_eligible,
    _make_tobit_obj, _XGB_BASE,
)

P0 = 0.0005
Q_FLOOR = 0.0005
N_INNER = 4
KF_COL = "kf_market_value"


# ---------------------------------------------------------------------------
# Champion pipeline, split into fit / predict
# ---------------------------------------------------------------------------

def fit_champion_models(train: pd.DataFrame, features: list[str],
                        clf_features: list[str], seed: int,
                        sigma: float = 0.02, gate_frac: float = 0.55,
                        floor_gate_k: float = 2.0):
    """Fit the champion Stage-1 Grabit + route classifier on one slice.

    Byte-for-byte the training path of route_mixture.grabit_latent (shipped
    defaults: censor_c=None, sigma_left=None) plus make_champion_fitter's
    classifier fit — split into fit and predict so ONE fitted pair can price
    several frames (the inner-val Year-1 rows AND the df_full intermediate
    rows) without refitting.
    """
    y_tr = train[TARGET].values
    max_elig_tr = train["max_eligible_pct"].values
    is_max_tr = train["is_max_contract"].values.astype(bool)

    base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
    base.fit(train[features], y_tr)
    bp = base.predict(train[features])
    gate = is_max_tr & (bp >= gate_frac * max_elig_tr)
    if floor_gate_k > 0 and "is_at_floor" in train.columns:
        gate_l = train["is_at_floor"].values & (bp <= floor_gate_k * y_tr)
    else:
        gate_l = np.zeros(len(train), bool)

    model = XGBRegressor(**{**_XGB_BASE, "random_state": seed,
                            "objective": _make_tobit_obj(gate, sigma,
                                                         left_mask=gate_l),
                            "base_score": float(y_tr.mean())})
    model.fit(train[features], y_tr)
    clf = train_route_classifier(train, clf_features, seed)
    return model, clf


def predict_champion(model, clf, test: pd.DataFrame, features: list[str],
                     clf_features: list[str]) -> np.ndarray:
    """The champion composition (push -> clip -> extension clip) on `test`.

    Identical to make_champion_fitter's output for the same fitted models. On
    df_full intermediate rows is_extension is False everywhere, so Stage 3 is
    a no-op there and the measurement is the ex-post market-value estimate.
    """
    latent = model.predict(test[features])
    lo = (test["floor_pct"].values if "floor_pct" in test.columns
          else np.zeros(len(test)))
    hi = test["max_eligible_pct"].values
    p_max = route_proba(clf, test, clf_features)[:, MAX_IDX]
    return compose(latent, lo=lo, hi=hi, p_max=p_max,
                   is_extension=test["is_extension"].values,
                   ext_cap_pct=test["ext_cap_pct"].values)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_full_frame(df_eval: pd.DataFrame, features: list[str]
                       ) -> pd.DataFrame:
    """load_training_data() made predictable by the champion pipeline.

    The intermediate rows only ever appear on the TEST side of a fit, so they
    need the prediction-time columns: the 21 features (median-filled with the
    evaluation frame's medians, the scale every model here was fit on), the
    Stage-2 bounds, inert Stage-3 route facts, and the classifier columns.
    """
    df_full = load_training_data()
    df_full = _compute_max_eligible(df_full)

    # Stage-2 lower bound: the evaluation frame's per-season floor. Exact
    # per-row floors need the experience bucket; for a KF measurement the
    # difference is noise, and grabit_latent would otherwise clip at zero.
    season_floor = df_eval.groupby("season")["floor_pct"].median()
    df_full["floor_pct"] = (df_full["season"].map(season_floor)
                            .fillna(float(df_eval["floor_pct"].min())))

    # Stage 3 is inert on intermediate rows: an escalator season is not a
    # first-paying-year extension, and the measurement wanted is the model's
    # market-value estimate, not a route-capped price.
    df_full["is_extension"] = False
    df_full["ext_cap_pct"] = np.nan

    med = df_eval[features].median()
    df_full[features] = df_full[features].fillna(med).fillna(0)

    # Classifier columns: base cols are a subset of `features` (already
    # filled); enrichment cols keep native NaN, their tested ship form.
    df_full, _ = attach_clf_features(df_full)
    return df_full.reset_index(drop=True)


def build_anchor_map(df_eval: pd.DataFrame, df_full: pd.DataFrame):
    """The three-tier anchor map (--anchor market). Per eval row:

        tier[i]        1 market anchor, 2 rookie anchor, 0 fallback (kf=prev)
        anchor_val[i]  the anchor cap_pct (NaN on tier 0)
        inter_idx[i]   df_full positions of the measurement seasons, in
                       chronological order (empty on tier 0 / no seasons)

    Tier-1 candidates are MARKET PRICING EVENTS, two kinds:
      - the player's own PRIOR rows in the eval frame — eval membership IS
        the project's definition of a fresh market price. Anchor value =
        the EVAL row's cap_pct (cap-hold normalized, salary-corrected).
      - prorated year-1 stints (raw label, cap_pct < PRORATED_FLOOR): a
        10-day at the minimum RATE is a market verdict ("worth the
        minimum") whose observed cap_pct merely prorates playing time, so
        the anchor value is PULLED UP to that season's full-season minimum
        cap charge (the eval frame's per-season floor_pct).
    Continuation mislabels, rookie scale and first contracts still cannot
    anchor — their seasons become measurements inside the window. Tier 2
    anchors at the earliest observed rookie-scale season (deal year 2) and
    is trusted like one measurement (P0 = R, see main).
    """
    from src.model.train import _load_rookie_scale_set, PRORATED_FLOOR
    rs = _load_rookie_scale_set()
    season_floor = df_eval.groupby("season")["floor_pct"].median().to_dict()
    floor_min = float(df_eval["floor_pct"].min())

    # market pricing events: player -> {season: (anchor value, is_prorated)}
    events: dict[str, dict[int, tuple[float, bool]]] = {}
    for p, s, c in zip(df_eval["player_name_norm"],
                       df_eval["season"].astype(int), df_eval[TARGET].values):
        events.setdefault(p, {}).setdefault(int(s), (float(c), False))

    full_cp: dict[tuple[str, int], float] = {}
    rows_by_player: dict[str, dict[int, int]] = {}
    yic = df_full["year_in_contract"] if "year_in_contract" in df_full.columns \
        else pd.Series(np.nan, index=df_full.index)
    for pos, (p, s, c, y1) in enumerate(zip(df_full["player_name_norm"],
                                            df_full["season"].astype(int),
                                            df_full[TARGET].values,
                                            yic.values)):
        if (p, s) not in full_cp:         # duplicates: keep the first row
            full_cp[(p, s)] = float(c)
            rows_by_player.setdefault(p, {})[s] = pos
            if y1 == 1 and c < PRORATED_FLOOR:
                events.setdefault(p, {}).setdefault(
                    s, (season_floor.get(s, floor_min), True))
    rookie_seasons: dict[str, list[int]] = {}
    for (p, s) in rs:
        if (p, s) in full_cp:
            rookie_seasons.setdefault(p, []).append(s)

    tier = np.zeros(len(df_eval), dtype=int)
    anchor_val = np.full(len(df_eval), np.nan)
    anchor_prorated = np.zeros(len(df_eval), dtype=bool)
    inter_idx, needed = [], set()
    for i, (p, T) in enumerate(zip(df_eval["player_name_norm"],
                                   df_eval["season"].astype(int))):
        market = [s for s in events.get(p, ()) if s < T]
        if market:
            t0 = max(market)
            tier[i] = 1
            anchor_val[i], anchor_prorated[i] = events[p][t0]
        else:
            rook = [s for s in rookie_seasons.get(p, ()) if s < T]
            if not rook:
                inter_idx.append([])
                continue
            t0 = min(rook)                # earliest observed = deal year 2
            tier[i] = 2
            anchor_val[i] = full_cp[(p, t0)]
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), tier, anchor_val, anchor_prorated


def build_intermediate_map_raw(df_eval: pd.DataFrame, df_full: pd.DataFrame):
    """Variant A's window builder (--anchor prev), kept for reproducibility.

    T0 is the player's most recent prior year_in_contract==1 season in
    df_full — the RAW label, dirty anchors included (52 prorated stints, 132
    rookie-scale rows, 197 continuations/first contracts of the 716 anchors).
    Returns (inter_idx, needed_idx, n_anchor, t0_pos).
    """
    year1: dict[str, list[int]] = {}
    year1_pos: dict[tuple[str, int], int] = {}
    yic = df_full["year_in_contract"] if "year_in_contract" in df_full.columns \
        else pd.Series(np.nan, index=df_full.index)
    rows_by_player: dict[str, dict[int, int]] = {}
    seen = set()
    for pos, (p, s, y1) in enumerate(zip(df_full["player_name_norm"],
                                         df_full["season"].astype(int),
                                         yic.values)):
        if (p, s) not in seen:            # duplicates (mid-season buyouts)
            seen.add((p, s))              # keep the first row per (p, s)
            rows_by_player.setdefault(p, {})[s] = pos
        if y1 == 1:
            year1.setdefault(p, []).append(s)
            year1_pos.setdefault((p, s), pos)

    inter_idx, needed = [], set()
    t0_pos = np.full(len(df_eval), -1, dtype=int)
    n_anchor = 0
    for i, (p, T) in enumerate(zip(df_eval["player_name_norm"],
                                   df_eval["season"].astype(int))):
        prior = [s for s in year1.get(p, ()) if s < T]
        if not prior:
            inter_idx.append([])
            continue
        n_anchor += 1
        t0 = max(prior)
        t0_pos[i] = year1_pos[(p, t0)]
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), n_anchor, t0_pos


def estimate_q(df_full: pd.DataFrame, players: set, r_var: float
               ) -> tuple[float, float, int]:
    """Process noise from year-over-year dcap_pct of multi-contract players.

    Restricted to the OUTER TRAINING players so no test-fold target informs
    the constant. Under a random walk observed with variance R,
    var(y_t - y_{t-1}) = Q + 2R, hence the subtraction; the floor keeps the
    filter from freezing when escalator years make the raw variance small.
    """
    sub = df_full[df_full["player_name_norm"].isin(players)]
    if "year_in_contract" in sub.columns:
        counts = (sub[sub["year_in_contract"] == 1]
                  .groupby("player_name_norm").size())
        multi = set(counts[counts >= 2].index)
        sub = sub[sub["player_name_norm"].isin(multi)]
    sub = (sub.drop_duplicates(["player_name_norm", "season"])
           .sort_values(["player_name_norm", "season"]))
    deltas = []
    for _, g in sub.groupby("player_name_norm"):
        s = g["season"].values.astype(int)
        v = g[TARGET].values
        consec = s[1:] == s[:-1] + 1
        deltas.extend((v[1:] - v[:-1])[consec])
    var_d = float(np.var(deltas, ddof=1)) if len(deltas) > 2 else 0.0
    return max(var_d - 2.0 * r_var, Q_FLOOR), var_d, len(deltas)


# ---------------------------------------------------------------------------
# The Kalman filter
# ---------------------------------------------------------------------------

def kalman_update(anchor: float, measurements, q: float, r: float,
                  p0: float = P0) -> float:
    """Random-walk KF: anchor at prev_cap_pct, one update per season."""
    x, p = float(anchor), p0
    for z in measurements:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x + k * (float(z) - x)
        p = (1.0 - k) * p_pred
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of seeds (10 = the canonical gate)")
    ap.add_argument("--anchor", choices=("prev", "market"), default="market",
                    help="'market' = three-tier anchor (eval-frame market "
                         "price / rookie deal-year-2 with P0=R / fallback to "
                         "prev_cap_pct); 'prev' = variant A, raw-label "
                         "windows anchored at prev_cap_pct, kept only for "
                         "reproducibility")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t_start = time.time()

    line = "=" * 70
    print(line)
    print("NESTED-CV kf_market_value ABLATION (full-model measurement)")
    print(f"anchor mode: {args.anchor} "
          f"({'three-tier: market price / rookie y2 with P0=R / prev fallback' if args.anchor == 'market' else 'variant A: prev_cap_pct, raw-label windows'})")
    print(line)

    df_eval, features = load_evaluation_frame()
    df_eval, clf_features = attach_clf_features(df_eval)
    assert set(features) == set(FEATURE_COLS), (
        f"evaluation frame features != FEATURE_COLS: "
        f"{set(FEATURE_COLS) ^ set(features)}")
    y = df_eval[TARGET].values
    groups = df_eval["player_name_norm"].values
    sel = ~df_eval["is_confirmation"].values
    prev = df_eval["prev_cap_pct"].values
    seas_all = df_eval["season"].values.astype(int)

    print(f"\nPreparing the full frame (all year_in_contract rows)...")
    df_full = prepare_full_frame(df_eval, features)
    if args.anchor == "market":
        (inter_idx, needed_idx, tier, anchor_val,
         anchor_prorated) = build_anchor_map(df_eval, df_full)
        anchor = np.where(tier > 0, anchor_val, prev)
        n_anchor = int((tier > 0).sum())
    else:
        inter_idx, needed_idx, n_anchor, _t0 = build_intermediate_map_raw(
            df_eval, df_full)
        tier = np.ones(len(df_eval), dtype=int)   # P0 everywhere (variant A)
        anchor_prorated = np.zeros(len(df_eval), dtype=bool)
        anchor = prev.copy()
    pos_of = {fi: k for k, fi in enumerate(needed_idx)}
    inter_pos = [[pos_of[fi] for fi in lst] for lst in inter_idx]
    full_needed = df_full.iloc[needed_idx]

    n_inter = sum(1 for lst in inter_idx if lst)
    counts = pd.Series([len(lst) for lst in inter_idx])
    print(f"eval frame {len(df_eval)} rows x {len(features)} features "
          f"({len(clf_features)} classifier features, frozen — "
          f"{KF_COL} joins the regression only)")
    print(f"full frame {len(df_full)} rows; intermediate predict subset "
          f"{len(needed_idx)} unique (player, season) rows")
    print(f"anchor: {n_anchor}/{len(df_eval)} rows anchored; {n_inter} have "
          f">=1 measurement season "
          f"(counts 1/2/3+: {int((counts == 1).sum())}/"
          f"{int((counts == 2).sum())}/{int((counts >= 3).sum())})")
    if args.anchor == "market":
        has_i = np.array([bool(lst) for lst in inter_idx])
        for t_id, lab in ((1, "tier 1 market-price anchor (P0=0.0005)"),
                          (2, "tier 2 rookie y2 anchor (P0=R)")):
            m = tier == t_id
            print(f"  {lab}: {int(m.sum())} rows, "
                  f"{int((m & has_i).sum())} with measurements")
        print(f"    of tier 1, prorated stints pulled to the season minimum: "
              f"{int(anchor_prorated.sum())}")
        print(f"  tier 3 fallback kf=prev_cap_pct: {int((tier == 0).sum())} rows")

    # Champion arm — the canonical protocol, 21 features. Run through
    # _fold_pass rather than oof_groupkfold so the per-(fold, seed)
    # predictions survive for Phase 4's fold-paired segment deltas;
    # _reduce_fold_pass then yields oof_groupkfold's exact outputs.
    n_seeds = len(seeds)
    print(f"\nChampion ({len(features)} features): GroupKFold over "
          f"{n_seeds} seeds x {N_SPLITS} folds...", flush=True)
    t0 = time.time()
    fitter_21 = make_champion_fitter(clf_features)
    store_c, folds_c, _ = _fold_pass(df_eval, features, fitter_21, seeds)
    oof_c, fr2_c, fr2sel_c = _reduce_fold_pass(df_eval, store_c, folds_c,
                                               seeds)[None]
    oof_c_by_seed = np.zeros((n_seeds, len(df_eval)))
    for rec in store_c:
        oof_c_by_seed[rec["si"], rec["va"]] = rec["pred"][None]
    print(f"  done in {time.time() - t0:.0f}s   "
          f"A1={r2_score(y, oof_c):.4f}", flush=True)

    # Candidate arms — nested CV, kf recomputed per outer fold. Two tests off
    # the same inner OOF and the same kf values:
    #   ADD   21 champion features + kf_market_value (22). Diluted by the
    #         kf <-> prev_cap_pct collinearity; kept as the reference the
    #         Phase-1 result (dSel +0.00045, t +0.61) was quoted on.
    #   SWAP  prev_cap_pct -> kf_market_value (21). The sharper test: kf IS a
    #         refinement of prev_cap_pct (identical wherever no intermediate
    #         season exists), so the hypothesis is "the refined price signal
    #         beats the raw one", not "the model wants both columns".
    assert "prev_cap_pct" in features
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df_eval, y, groups))
    features_add = list(features) + [KF_COL]
    features_swap = [KF_COL if f == "prev_cap_pct" else f for f in features]
    fitter_cand = make_champion_fitter(clf_features)

    oof_add_acc = np.zeros(len(df_eval))
    oof_swap_acc = np.zeros(len(df_eval))
    oof_ctrl_acc = np.zeros(len(df_eval))
    oof_add_by_seed = np.zeros((n_seeds, len(df_eval)))
    oof_swap_by_seed = np.zeros((n_seeds, len(df_eval)))
    fr2_add = np.zeros((N_SPLITS, n_seeds))
    fr2sel_add = np.zeros((N_SPLITS, n_seeds))
    fr2_swap = np.zeros((N_SPLITS, n_seeds))
    fr2sel_swap = np.zeros((N_SPLITS, n_seeds))
    fr2_ctrl = np.zeros((N_SPLITS, n_seeds))
    fr2sel_ctrl = np.zeros((N_SPLITS, n_seeds))
    kf_heldout = np.zeros((n_seeds, len(df_eval)))   # each row's own-fold kf
    inner_r2_by_seed = []
    r_by_seed, q_by_seed = [], []

    print(f"\nCandidates (ADD {len(features_add)} / SWAP {len(features_swap)} "
          f"features): nested CV, {N_INNER}-fold inner OOF per outer fold...",
          flush=True)
    for si, seed in enumerate(seeds):
        t_seed = time.time()
        inner_r2s, r_vals, q_vals = [], [], []
        for fi, (tr, va) in enumerate(folds):
            train_df = df_eval.iloc[tr]
            y_tr, g_tr = y[tr], groups[tr]

            # Inner OOF over the outer training rows (grouped by player, so
            # a player's every Year-1 row sits in one inner fold).
            inner_folds = list(GroupKFold(n_splits=N_INNER)
                               .split(train_df, y_tr, g_tr))
            inner_oof = np.full(len(tr), np.nan)
            full_pred = np.zeros((N_INNER, len(needed_idx)))
            player_inner: dict[str, int] = {}
            for j, (itr, iva) in enumerate(inner_folds):
                model, clf = fit_champion_models(train_df.iloc[itr], features,
                                                 clf_features, seed)
                inner_oof[iva] = predict_champion(model, clf,
                                                  train_df.iloc[iva],
                                                  features, clf_features)
                full_pred[j] = predict_champion(model, clf, full_needed,
                                                features, clf_features)
                for p in np.unique(g_tr[iva]):
                    player_inner[p] = j
            full_mean = full_pred.mean(axis=0)

            resid = y_tr - inner_oof
            r_var = float(np.var(resid, ddof=1))
            q_var, _, _ = estimate_q(df_full, set(g_tr), r_var)
            inner_r2s.append(r2_score(y_tr, inner_oof))
            r_vals.append(r_var)
            q_vals.append(q_var)

            # kf for this fold's train AND test rows, from this fold's inner
            # world only.
            def kf_rows(rows):
                out = np.full(len(rows), np.nan)
                for k, i in enumerate(rows):
                    if not np.isfinite(anchor[i]):
                        continue                      # -> median impute below
                    pos_list = inter_pos[i]
                    if not pos_list:
                        out[k] = anchor[i]            # no intermediate seasons
                        continue
                    j = player_inner.get(groups[i])
                    zs = (full_pred[j, pos_list] if j is not None
                          else full_mean[pos_list])
                    # tier-2 anchors are slotted prices, not market judgments:
                    # their prior variance is R (one measurement's trust).
                    p0 = r_var if tier[i] == 2 else P0
                    out[k] = kalman_update(anchor[i], zs, q_var, r_var, p0)
                return out

            kf_tr, kf_va = kf_rows(tr), kf_rows(va)
            kf_heldout[si, va] = np.where(np.isfinite(kf_va), kf_va, prev[va])

            train_22 = train_df.copy()
            test_22 = df_eval.iloc[va].copy()
            train_22[KF_COL] = kf_tr
            test_22[KF_COL] = kf_va
            fill = float(np.nanmedian(kf_tr)) if np.isfinite(kf_tr).any() \
                else float(np.nanmedian(prev))
            train_22[KF_COL] = train_22[KF_COL].fillna(fill)
            test_22[KF_COL] = test_22[KF_COL].fillna(fill)

            vs = sel[va]
            pred_add = fitter_cand(train_22, test_22, features_add, seed)
            oof_add_acc[va] += pred_add
            oof_add_by_seed[si, va] = pred_add
            fr2_add[fi, si] = r2_score(y[va], pred_add)
            fr2sel_add[fi, si] = (r2_score(y[va][vs], pred_add[vs])
                                  if vs.sum() > 10 else np.nan)
            pred_swap = fitter_cand(train_22, test_22, features_swap, seed)
            oof_swap_acc[va] += pred_swap
            oof_swap_by_seed[si, va] = pred_swap
            fr2_swap[fi, si] = r2_score(y[va], pred_swap)
            fr2sel_swap[fi, si] = (r2_score(y[va][vs], pred_swap[vs])
                                   if vs.sum() > 10 else np.nan)

            # Season-dummy control: prev + the TRAIN fold's per-season mean of
            # (kf - prev). Same season/coverage structure as kf, zero
            # player-level content — the supply_samepos is2019 lesson: a
            # feature whose values align with seasons must beat this arm,
            # not just the champion.
            dkf_tr = train_22[KF_COL].values - prev[tr]
            sm = pd.Series(dkf_tr).groupby(seas_all[tr]).mean().to_dict()
            train_23 = train_df.copy()
            test_23 = df_eval.iloc[va].copy()
            train_23[KF_COL] = prev[tr] + np.array(
                [sm.get(s, 0.0) for s in seas_all[tr]])
            test_23[KF_COL] = prev[va] + np.array(
                [sm.get(s, 0.0) for s in seas_all[va]])
            pred_ctrl = fitter_cand(train_23, test_23, features_swap, seed)
            oof_ctrl_acc[va] += pred_ctrl
            fr2_ctrl[fi, si] = r2_score(y[va], pred_ctrl)
            fr2sel_ctrl[fi, si] = (r2_score(y[va][vs], pred_ctrl[vs])
                                   if vs.sum() > 10 else np.nan)

        inner_r2_by_seed.append(float(np.mean(inner_r2s)))
        r_by_seed.append(float(np.mean(r_vals)))
        q_by_seed.append(float(np.mean(q_vals)))
        print(f"  seed {seed}: inner OOF R2={inner_r2_by_seed[-1]:.4f}  "
              f"R={r_by_seed[-1]:.5f}  Q={q_by_seed[-1]:.5f}  "
              f"add [{' '.join(f'{fr2_add[fi, si]:.3f}' for fi in range(N_SPLITS))}]  "
              f"swap [{' '.join(f'{fr2_swap[fi, si]:.3f}' for fi in range(N_SPLITS))}]  "
              f"{time.time() - t_seed:.0f}s", flush=True)

    oof_add = oof_add_acc / n_seeds
    oof_swap = oof_swap_acc / n_seeds
    oof_ctrl = oof_ctrl_acc / n_seeds
    kf_mean = kf_heldout.mean(axis=0)

    # ------------------------------------------------------------------ report
    print(f"\n{line}")
    print("Phase 1 — Inner OOF accuracy (the KF measurement's quality)")
    print(line)
    for si, seed in enumerate(seeds):
        print(f"  Seed {seed}: inner OOF R2 = {inner_r2_by_seed[si]:.4f}")
    print(f"  Mean inner OOF R2: {np.mean(inner_r2_by_seed):.4f}  "
          f"(simple predictor was 0.46)")
    print(f"  Mean R (measurement var): {np.mean(r_by_seed):.5f}  "
          f"(simple predictor was ~0.0037)")
    print(f"  Mean Q (process var):     {np.mean(q_by_seed):.5f}  "
          f"(floor {Q_FLOOR})")

    moved = np.abs(kf_mean - prev)
    has_inter = np.array([bool(lst) for lst in inter_idx])
    print(f"\n{line}")
    print("Phase 2 — KF diagnostic (held-out kf values, seed-averaged)")
    print(line)
    if args.anchor == "market":
        has_a = tier > 0
        print(f"  anchor <-> prev_cap_pct:  "
              f"r={np.corrcoef(anchor, prev)[0, 1]:.3f}   "
              f"|anchor - prev| mean={np.abs(anchor - prev)[has_a].mean():.4f} "
              f"(over the {int(has_a.sum())} anchored rows)")
    print(f"  {KF_COL} <-> prev_cap_pct:  r={np.corrcoef(kf_mean, prev)[0, 1]:.3f}")
    print(f"  {KF_COL} <-> {TARGET}:       r={np.corrcoef(kf_mean, y)[0, 1]:.3f}"
          f"   (prev_cap_pct <-> {TARGET}: r={np.corrcoef(prev, y)[0, 1]:.3f})")
    print(f"  |kf - prev_cap_pct|: mean={moved.mean():.4f}  "
          f"median={np.median(moved):.4f}  "
          f"(rows with intermediates only: mean={moved[has_inter].mean():.4f})")
    print(f"  Rows moved by >0.1% of cap: {int((moved > 0.001).sum())}"
          f"/{len(df_eval)}")
    if args.anchor == "market":
        for t_id, lab in ((1, "tier 1 (market anchor)"),
                          (2, "tier 2 (rookie anchor)")):
            m = tier == t_id
            if m.any():
                print(f"  {lab}: n={int(m.sum())}  "
                      f"|kf-prev| mean={moved[m].mean():.4f}  "
                      f"moved>0.1%cap {int((moved[m] > 0.001).sum())}  "
                      f"r(kf,y)={np.corrcoef(kf_mean[m], y[m])[0, 1]:.3f} "
                      f"(r(prev,y)={np.corrcoef(prev[m], y[m])[0, 1]:.3f})")

    recent = df_eval["season"].values >= 2024
    mae_c, _ = _dollars(df_eval, oof_c)

    def _arm_line(tag, oof_x, n_feat):
        mae_x, _ = _dollars(df_eval, oof_x)
        print(f"  {tag:28s} A1={r2_score(y, oof_x):.4f}  "
              f"sel={r2_score(y[sel], oof_x[sel]):.4f}  "
              f"A2={r2_score(y[recent], oof_x[recent]):.4f}  "
              f"MAE=${mae_x:.2f}M  ({n_feat} features)")
        return mae_x

    def _delta_lines(fr2sel_x, fr2_x):
        d_sel = paired_delta(fr2sel_c, fr2sel_x)
        d_pool = paired_delta(fr2_c, fr2_x)
        ok = d_sel["t"] > 2
        print(f"    dSel={d_sel['delta']:+.5f} +/- {d_sel['se']:.5f}  "
              f"t={d_sel['t']:+.2f}  [{'PASS' if ok else 'FAIL'}]  "
              f"dPooled={d_pool['delta']:+.5f} t={d_pool['t']:+.2f}")
        print(f"    folds (selection): "
              + " ".join(f"{v:+.5f}" for v in d_sel["per_fold"]))
        return d_sel, d_pool, ok

    print(f"\n{line}")
    print("Phase 3 — Ablation (paired, identical folds and seeds; "
          "gate: selection-pool t > 2)")
    print(line)
    _arm_line("Champion (prev_cap_pct)", oof_c, len(features))
    mae_add = _arm_line(f"ADD  + {KF_COL}", oof_add, len(features_add))
    d_sel_add, d_pool_add, pass_add = _delta_lines(fr2sel_add, fr2_add)
    mae_swap = _arm_line(f"SWAP prev_cap_pct -> kf", oof_swap,
                         len(features_swap))
    d_sel_swap, d_pool_swap, pass_swap = _delta_lines(fr2sel_swap, fr2_swap)
    mae_ctrl = _arm_line("CTRL season-dummy", oof_ctrl, len(features_swap))
    d_sel_ctrl = paired_delta(fr2sel_c, fr2sel_ctrl)
    d_pool_ctrl = paired_delta(fr2_c, fr2_ctrl)
    share = (d_sel_ctrl["delta"] / d_sel_swap["delta"]
             if d_sel_swap["delta"] else float("nan"))
    print(f"    dSel={d_sel_ctrl['delta']:+.5f} t={d_sel_ctrl['t']:+.2f}  "
          f"dPooled={d_pool_ctrl['delta']:+.5f}")
    print(f"    season-structure-only arm recovers {share:.0%} of the swap "
          f"gain (is2019 recovered 70% of supply_samepos -> reject; the swap "
          f"must clearly beat this arm)")

    # Phase 4 — the 2x3 view: champion (prev_cap_pct) vs SWAP (kf) on all
    # rows, on the tier-3 fallback segment (kf == prev by construction — any
    # movement here is collateral from the shared model, not from the
    # feature's own values), and on the kf-active segment where the feature
    # actually differs. Fold-paired deltas; a segment R2 cell needs >= 10
    # validation rows.
    seg_json = {}
    if args.anchor == "market":
        cap_m = df_eval["cap"].values / 1e6

        def seg_stats(m):
            mat_c = np.full((N_SPLITS, n_seeds), np.nan)
            mat_s = np.full((N_SPLITS, n_seeds), np.nan)
            for fi, (_tr, va) in enumerate(folds):
                vm = va[m[va]]
                if len(vm) < 10:
                    continue
                for si in range(n_seeds):
                    mat_c[fi, si] = r2_score(y[vm], oof_c_by_seed[si, vm])
                    mat_s[fi, si] = r2_score(y[vm], oof_swap_by_seed[si, vm])
            ok = ~np.isnan(mat_c).any(axis=1)
            d = paired_delta(mat_c[ok], mat_s[ok]) if ok.sum() >= 3 else None
            return {
                "n": int(m.sum()),
                "champ_r2": float(r2_score(y[m], oof_c[m])),
                "swap_r2": float(r2_score(y[m], oof_swap[m])),
                "champ_mae_m": float(np.abs((oof_c[m] - y[m]) * cap_m[m]).mean()),
                "swap_mae_m": float(np.abs((oof_swap[m] - y[m]) * cap_m[m]).mean()),
                "paired": d, "n_folds_scored": int(ok.sum()),
            }

        print(f"\n{line}")
        print("Phase 4 — kf vs prev_cap_pct by segment "
              "(champion vs SWAP, fold-paired)")
        print(line)
        for lab, m in (("all rows", np.ones(len(df_eval), bool)),
                       ("tier 3: kf=prev fallback", tier == 0),
                       ("kf-active: tier 1+2", tier > 0)):
            s = seg_stats(m)
            d = s["paired"]
            dtxt = (f"dR2={d['delta']:+.5f} t={d['t']:+.2f} "
                    f"({s['n_folds_scored']} folds)" if d else "n too small")
            print(f"  {lab:28s} n={s['n']:3d}  "
                  f"R2 {s['champ_r2']:.4f} -> {s['swap_r2']:.4f}  {dtxt}")
            print(f"  {'':28s} MAE ${s['champ_mae_m']:.2f}M -> "
                  f"${s['swap_mae_m']:.2f}M "
                  f"({s['swap_mae_m'] - s['champ_mae_m']:+.2f})")
            seg_json[lab] = s

    # Phase 5 — C-layer guards for the SWAP candidate (evaluate_suite rules):
    # C1 relative calibration gate |slope-1| excess <= 0.005; C2 no signing
    # mechanism's |bias| may grow by more than $0.3M.
    cap_m_all = df_eval["cap"].values / 1e6
    slope_c = float(np.polyfit(oof_c, y, 1)[0])
    slope_s = float(np.polyfit(oof_swap, y, 1)[0])
    c1_excess = abs(slope_s - 1.0) - abs(slope_c - 1.0)
    print(f"\n{line}")
    print("Phase 5 — C-layer guards (champion vs SWAP)")
    print(line)
    print(f"  C1 calibration slope {slope_c:.4f} -> {slope_s:.4f}   "
          f"|slope-1| excess {c1_excess:+.4f}  "
          f"[{'PASS' if c1_excess <= 0.005 else 'FAIL'}] (gate <= +0.005)")
    c2_json, worst_cat, worst_growth = {}, None, -np.inf
    for cat, sub in df_eval.groupby("signing_cat"):
        m = (df_eval["signing_cat"] == cat).values
        if m.sum() < 10:
            continue
        err_c = (oof_c[m] - y[m]) * cap_m_all[m]
        err_s = (oof_swap[m] - y[m]) * cap_m_all[m]
        growth = abs(float(err_s.mean())) - abs(float(err_c.mean()))
        c2_json[str(cat)] = {
            "n": int(m.sum()), "champ_bias_m": float(err_c.mean()),
            "swap_bias_m": float(err_s.mean()), "abs_bias_growth_m": growth,
            "champ_mae_m": float(np.abs(err_c).mean()),
            "swap_mae_m": float(np.abs(err_s).mean())}
        if growth > worst_growth:
            worst_cat, worst_growth = str(cat), growth
    print(f"  C2 bias by signing mechanism (|bias| growth, gate <= +$0.3M):")
    for cat, d in sorted(c2_json.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {cat:14s} n={d['n']:4d}  bias ${d['champ_bias_m']:+5.2f}M "
              f"-> ${d['swap_bias_m']:+5.2f}M   growth "
              f"{d['abs_bias_growth_m']:+5.2f}")
    print(f"  worst: {worst_cat} {worst_growth:+.2f}M  "
          f"[{'PASS' if worst_growth <= 0.3 else 'FAIL'}]")

    out_dir = OUTPUTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(df_eval), "seeds": seeds, "n_inner": N_INNER,
        "anchor_mode": args.anchor,
        "p0": P0, "q_floor": Q_FLOOR,
        "anchor_rows": n_anchor, "rows_with_intermediates": int(n_inter),
        "tiers": {"tier1": int((tier == 1).sum()),
                  "tier1_prorated_min": int(anchor_prorated.sum()),
                  "tier2": int((tier == 2).sum()),
                  "tier3": int((tier == 0).sum())},
        "inner_oof_r2_by_seed": inner_r2_by_seed,
        "r_by_seed": r_by_seed, "q_by_seed": q_by_seed,
        "kf_diag": {
            "corr_kf_prev": float(np.corrcoef(kf_mean, prev)[0, 1]),
            "corr_kf_y": float(np.corrcoef(kf_mean, y)[0, 1]),
            "corr_prev_y": float(np.corrcoef(prev, y)[0, 1]),
            "abs_move_mean": float(moved.mean()),
            "abs_move_median": float(np.median(moved)),
            "rows_moved_gt_0p001": int((moved > 0.001).sum()),
        },
        "champion": {"a1": float(r2_score(y, oof_c)),
                     "sel": float(r2_score(y[sel], oof_c[sel])),
                     "a2": float(r2_score(y[recent], oof_c[recent])),
                     "mae_m": mae_c},
        "candidate_add": {"a1": float(r2_score(y, oof_add)),
                          "sel": float(r2_score(y[sel], oof_add[sel])),
                          "a2": float(r2_score(y[recent], oof_add[recent])),
                          "mae_m": mae_add,
                          "paired_selection": d_sel_add,
                          "paired_pooled": d_pool_add,
                          "gate_pass": bool(pass_add)},
        "candidate_swap": {"a1": float(r2_score(y, oof_swap)),
                           "sel": float(r2_score(y[sel], oof_swap[sel])),
                           "a2": float(r2_score(y[recent], oof_swap[recent])),
                           "mae_m": mae_swap,
                           "paired_selection": d_sel_swap,
                           "paired_pooled": d_pool_swap,
                           "gate_pass": bool(pass_swap)},
        "candidate_control": {"a1": float(r2_score(y, oof_ctrl)),
                              "sel": float(r2_score(y[sel], oof_ctrl[sel])),
                              "mae_m": mae_ctrl,
                              "paired_selection": d_sel_ctrl,
                              "paired_pooled": d_pool_ctrl,
                              "share_of_swap_gain": float(share)},
        "c_layer": {"c1_slope_champ": slope_c, "c1_slope_swap": slope_s,
                    "c1_excess": float(c1_excess),
                    "c1_pass": bool(c1_excess <= 0.005),
                    "c2_by_mechanism": c2_json,
                    "c2_worst": worst_cat,
                    "c2_worst_growth_m": float(worst_growth),
                    "c2_pass": bool(worst_growth <= 0.3)},
        "segments_kf_vs_prev": seg_json,
        "fold_r2_selection": {"champion": fr2sel_c.tolist(),
                              "candidate_add": fr2sel_add.tolist(),
                              "candidate_swap": fr2sel_swap.tolist()},
        "fold_r2": {"champion": fr2_c.tolist(),
                    "candidate_add": fr2_add.tolist(),
                    "candidate_swap": fr2_swap.tolist()},
        "runtime_s": round(time.time() - t_start, 1),
    }
    out = out_dir / f"ablation_kf_market_value_full_{args.anchor}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ref = df_eval[["player_name_norm", "season", TARGET, "signing_cat",
                   "is_confirmation"]].copy()
    ref["tier"] = tier
    ref["prev_cap_pct"] = prev
    ref["kf_mean_heldout"] = kf_mean
    ref["oof_champion"] = oof_c
    ref["oof_add"] = oof_add
    ref["oof_swap"] = oof_swap
    ref["oof_control"] = oof_ctrl
    ref_path = out_dir / f"ablation_kf_oof_{args.anchor}.csv"
    ref.to_csv(ref_path, index=False)
    print(f"\nSaved {out}")
    print(f"Saved {ref_path} ({len(ref)} rows)   "
          f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
