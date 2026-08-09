"""kf_market_value: Kalman-filtered salary trajectory from market anchors.

For a player at season T, anchors at the market's last observed price and
updates through model-predicted intermediate seasons (escalator years).
Three-tier anchor: (1) most recent Year-1 eval-frame row, (2) earliest
rookie-scale season for first-rounders with no market anchor, (3) fallback
to prev_cap_pct. Adopted as a SWAP for prev_cap_pct in v8.13x.

Training-time values are fold-honest (nested CV in the ablation harness);
inference-time values use the base model (21 features with prev_cap_pct)
to price intermediate seasons — no circularity because the measurement
model never sees kf_market_value.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.model.train import TARGET, PRORATED_FLOOR, _load_rookie_scale_set

P0 = 0.0005
Q_FLOOR = 0.0005

MEASUREMENT_FEATURES = [
    "darko_dpm_z", "lebron_z", "laker_z",
    "age", "age_squared",
    "mpg",
    "availability_3yr",
    "is_waived",
    "usage_pct",
    "height_inches",
    "cba_era",
    "ast_pct",
    "award_score_cum",
    "draft_pick",
    "prev_cap_pct",
    "mpg_x_waived",
    "playoff_mpg_diff",
    "kalman_filtered_stats",
    "darko_od_diff_z",
    "lebron_od_diff_z",
    "laker_od_diff_z",
]


def kalman_update(anchor, measurements, q, r, p0=P0):
    """Random-walk KF: anchor at market price, one update per season."""
    x, p = float(anchor), p0
    for z in measurements:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x + k * (float(z) - x)
        p = (1.0 - k) * p_pred
    return x


def estimate_q(df_full, players, r_var):
    """Process noise from year-over-year Δcap_pct of multi-contract players."""
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
    return max(var_d - 2.0 * r_var, Q_FLOOR)


def build_anchor_map(df_eval, df_full):
    """Three-tier anchor map.

    Returns (inter_idx, needed_idx, tier, anchor_val):
        inter_idx:  list of lists — df_full positions for intermediate seasons
        needed_idx: sorted unique df_full positions that need prediction
        tier:       int array (0=fallback, 1=market anchor, 2=rookie anchor)
        anchor_val: float array of anchor cap_pct values
    """
    rs = _load_rookie_scale_set()
    season_floor = df_eval.groupby("season")["floor_pct"].median().to_dict()
    floor_min = float(df_eval["floor_pct"].min())

    events: dict[str, dict[int, tuple[float, bool]]] = {}
    for p, s, c in zip(df_eval["player_name_norm"],
                       df_eval["season"].astype(int), df_eval[TARGET].values):
        events.setdefault(p, {}).setdefault(int(s), (float(c), False))

    full_cp: dict[tuple[str, int], float] = {}
    rows_by_player: dict[str, dict[int, int]] = {}
    yic = (df_full["year_in_contract"] if "year_in_contract" in df_full.columns
           else pd.Series(np.nan, index=df_full.index))
    for pos, (p, s, c, y1) in enumerate(zip(
            df_full["player_name_norm"], df_full["season"].astype(int),
            df_full[TARGET].values, yic.values)):
        if (p, s) not in full_cp:
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
    inter_idx, needed = [], set()
    for i, (p, T) in enumerate(zip(df_eval["player_name_norm"],
                                   df_eval["season"].astype(int))):
        market = [s for s in events.get(p, ()) if s < T]
        if market:
            t0 = max(market)
            tier[i] = 1
            anchor_val[i] = events[p][t0][0]
        else:
            rook = [s for s in rookie_seasons.get(p, ()) if s < T]
            if not rook:
                inter_idx.append([])
                continue
            t0 = min(rook)
            tier[i] = 2
            anchor_val[i] = full_cp[(p, t0)]
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), tier, anchor_val


def compute_kf_column(df_eval, df_full, predict_fn, r_var,
                      players=None):
    """Compute kf_market_value for every row in df_eval.

    Args:
        df_eval:    filtered Year-1 frame (rows to compute kf for)
        df_full:    full dataset (all year_in_contract, for intermediate seasons)
        predict_fn: callable(df_subset) → predictions array.
                    Used to price intermediate seasons in df_full.
        r_var:      measurement noise variance (from base model's OOF residuals
                    or in-sample residuals)
        players:    set of player names for Q estimation (default: all in df_eval)

    Returns:
        kf_values: array of kf_market_value, one per df_eval row.
                   NaN where no anchor exists.
    """
    inter_idx, needed_idx, tier, anchor_val = build_anchor_map(df_eval, df_full)

    if players is None:
        players = set(df_eval["player_name_norm"].unique())
    q = estimate_q(df_full, players, r_var)

    prev = df_eval["prev_cap_pct"].values
    anchor = np.where(tier > 0, anchor_val, prev)

    pos_of = {fi: k for k, fi in enumerate(needed_idx)}
    inter_pos = [[pos_of[fi] for fi in lst] for lst in inter_idx]

    if needed_idx:
        full_needed = df_full.iloc[needed_idx]
        measurements = predict_fn(full_needed)
    else:
        measurements = np.array([])

    kf_values = np.full(len(df_eval), np.nan)
    for i in range(len(df_eval)):
        if not np.isfinite(anchor[i]):
            continue
        pos_list = inter_pos[i]
        if not pos_list:
            kf_values[i] = anchor[i]
            continue
        zs = measurements[[p for p in pos_list]]
        p0 = r_var if tier[i] == 2 else P0
        kf_values[i] = kalman_update(anchor[i], zs, q, r_var, p0)

    kf_values = np.where(np.isfinite(kf_values), kf_values, prev)
    return kf_values


def attach_kf_inference(df, df_full, base_model, base_features, base_medians,
                        clf, clf_features):
    """Compute kf_market_value at inference time using a fitted base model.

    The base model uses MEASUREMENT_FEATURES (prev_cap_pct, not kf). Its
    predictions on intermediate seasons become the KF measurements.

    Args:
        df:             the frame to attach kf_market_value to (must have
                        floor_pct and the eval-frame columns for anchor building)
        df_full:        full dataset for intermediate seasons
        base_model:     fitted XGBRegressor (21 features with prev_cap_pct)
        base_features:  feature list the base_model was trained on
        base_medians:   median fill values from the base model's training set
        clf:            fitted route classifier (for compose)
        clf_features:   classifier feature list

    Returns:
        df with kf_market_value column added.
    """
    from src.model.route_mixture import route_proba, MAX_IDX
    from src.model.stages import compose
    from src.model.train import _compute_max_eligible

    df_full_pred = df_full.copy()
    df_full_pred = _compute_max_eligible(df_full_pred)
    if "floor_pct" not in df_full_pred.columns:
        season_floor = df.groupby("season")["floor_pct"].median()
        df_full_pred["floor_pct"] = (df_full_pred["season"].map(season_floor)
                                     .fillna(float(df["floor_pct"].min())))
    df_full_pred["is_extension"] = False
    df_full_pred["ext_cap_pct"] = np.nan
    df_full_pred[base_features] = (df_full_pred[base_features]
                                   .fillna(base_medians).fillna(0))
    from src.model.route_mixture import attach_clf_features
    df_full_pred, _ = attach_clf_features(df_full_pred)

    def predict_fn(subset):
        X = subset[base_features].fillna(base_medians).fillna(0)
        latent = base_model.predict(X)
        lo = (subset["floor_pct"].values if "floor_pct" in subset.columns
              else np.zeros(len(subset)))
        hi = subset["max_eligible_pct"].values
        p_max = route_proba(clf, subset, clf_features)[:, MAX_IDX]
        return compose(latent, lo=lo, hi=hi, p_max=p_max,
                       is_extension=np.zeros(len(subset), bool),
                       ext_cap_pct=np.full(len(subset), np.nan))

    y_train = df[TARGET].values
    in_sample = base_model.predict(df[base_features].fillna(base_medians).fillna(0))
    r_var = float(np.var(y_train - in_sample, ddof=1))

    kf = compute_kf_column(df, df_full_pred, predict_fn, r_var)
    df = df.copy()
    df["kf_market_value"] = kf
    return df
