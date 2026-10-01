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

from config import CAP_BY_SEASON, PROCESSED_DIR
from src.model.train import TARGET, PRORATED_FLOOR, _load_rookie_scale_set, _load_debut_seasons

P0 = 0.0005
Q_FLOOR = 0.04

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


def load_prehistory_anchors() -> dict[str, dict[int, tuple[float, bool]]]:
    """Load pre-2019 Year-1 cap_pct values for KF anchor injection.

    Sources:
      - salaries_prehistory.csv: BBRef salary data for seasons 2016-2018
      - spotrac_signing_types.csv: Year-1 identification (season == contract_start)
      - Spotrac AAV for season 2015 where no BBRef salary exists

    Returns dict[player_name_norm -> {season: (cap_pct, is_prorated)}].
    Pre-2019 rows enter the anchor map only, never the training set.
    """
    # 1. Load Spotrac signing types to identify Year-1 contracts
    spotrac_path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not spotrac_path.exists():
        print("WARNING: spotrac_signing_types.csv not found, no prehistory anchors")
        return {}
    st = pd.read_csv(spotrac_path)
    st["season"] = st["season"].astype(int)
    st["contract_start"] = st["contract_start"].astype(int)
    y1 = st[(st["season"] == st["contract_start"])
            & (st["season"] >= 2015) & (st["season"] <= 2018)].copy()
    y1_set = set(zip(y1["player_name_norm"], y1["season"]))
    print(f"  Spotrac Year-1 contracts 2015-2018: {len(y1_set)}")

    # 2. Load BBRef prehistory salaries (2016-2018)
    prehistory_path = PROCESSED_DIR / "salaries_prehistory.csv"
    events: dict[str, dict[int, tuple[float, bool]]] = {}
    bbref_covered = set()
    if prehistory_path.exists():
        ph = pd.read_csv(prehistory_path)
        ph["season"] = ph["season"].astype(int)
        for _, r in ph.iterrows():
            p, s, sal = r["player_name_norm"], r["season"], r["salary"]
            if (p, s) not in y1_set:
                continue
            cap = CAP_BY_SEASON.get(s)
            if cap is None:
                continue
            cap_pct = float(sal) / float(cap)
            is_prorated = cap_pct < PRORATED_FLOOR
            events.setdefault(p, {})[s] = (cap_pct, is_prorated)
            bbref_covered.add((p, s))
        print(f"  BBRef prehistory Year-1 matches: {len(bbref_covered)}")

    # 3. Spotrac AAV fallback for 2015 and any BBRef gaps
    aav_used = 0
    for _, r in y1.iterrows():
        p, s = r["player_name_norm"], int(r["season"])
        if (p, s) in bbref_covered:
            continue
        cap = CAP_BY_SEASON.get(s)
        aav = r.get("aav")
        if cap is None or pd.isna(aav) or float(aav) <= 0:
            continue
        cap_pct = float(aav) / float(cap)
        is_prorated = cap_pct < PRORATED_FLOOR
        events.setdefault(p, {})[s] = (cap_pct, is_prorated)
        aav_used += 1
    print(f"  Spotrac AAV fallback anchors: {aav_used}")

    total = sum(len(v) for v in events.values())
    players = len(events)
    print(f"  Total prehistory anchors: {total} "
          f"({players} players, seasons 2015-2018)")
    return events


REPRICE_FLOOR = ("rest-of-season",)
REPRICE_ZERO = ("10-day", "two-way")


def load_reprice_events() -> dict[str, dict[int, tuple]]:
    """In-season signings: the market's later observed prices (KF anchor, v6.0.4).

    - A rest-of-season contract is the prorated minimum, so it anchors at the
      season floor.
    - A two-way contract counts 0 against the cap and anchors at 0 (weak).
    - A 10-day contract anchors at 0 (weak). This is the user's declared
      choice; its observed cap price is the prorated minimum.
    - A standard non-extension contract signed after opening night and before
      June 20 of the following year anchors at its own AAV over the cap. Later
      June signings start the next season, and anchoring on them would read
      a row's own contract.

    Returns player -> {season: (signing_date, kind, cap_pct or None)}, with
    kind in {"floor", "zero", "price"}, keeping each season's latest event.
    """
    from config import CAP_BY_SEASON
    from src.features.waiver_history import _season_opener
    sd = pd.read_csv(PROCESSED_DIR / "contract_signing_dates.csv",
                     parse_dates=["signing_date"])
    sd = sd.dropna(subset=["signing_date", "signing_season"])
    out: dict[str, dict[int, tuple]] = {}
    for r in sd.itertuples():
        s = int(r.signing_season)
        cls = str(r.contract_class).lower()
        value = None
        if cls in REPRICE_ZERO:
            kind = "zero"
        elif cls in REPRICE_FLOOR:
            kind = "floor"
        elif (cls == "standard" and not bool(r.is_extension)
              and _season_opener(s) < r.signing_date
              < pd.Timestamp(f"{s + 1}-06-20")):
            years = r.contract_years if pd.notna(r.contract_years) else 1
            cap = CAP_BY_SEASON.get(s)
            if pd.notna(r.total_value) and cap and years > 0:
                kind, value = "price", float(r.total_value) / years / cap
            else:
                kind = "floor"
        else:
            continue
        prev = out.setdefault(r.player_name_norm, {}).get(s)
        if prev is None or r.signing_date > prev[0]:
            out[r.player_name_norm][s] = (r.signing_date, kind, value)
    return out


def build_anchor_map(df_eval, df_full, extra_events=None,
                     expand_anchors=True, market_events=None,
                     reprice_events=None):
    """Three-tier anchor map.

    Args:
        df_eval: rows whose kf_market_value is being computed. At inference
            these can include escalator and rookie-scale rows that are not
            themselves fresh market prices.
        market_events: optional filtered Year-1 frame that supplies historical
            market anchors. Defaults to df_eval for backwards compatibility.
            Keeping this separate prevents an inference target from becoming
            its own market event while still exposing its prior signings.
        extra_events: optional pre-2019 Year-1 cap_pct anchors (from
            load_prehistory_anchors). Merged BEFORE eval-frame events so
            an eval-frame anchor at the same (player, season) wins.
        reprice_events: optional in-season re-pricing events
            (`load_reprice_events`). Each overrides that season's anchor at
            its observed price (floor, own AAV, or 0 with the tier-2 prior).
        expand_anchors: two tier-2 expansions (v8.14x default):
            (a) First-contract mirror: undrafted/2nd-round first contracts
                (exp <= 1, not rookie-scale) provide tier-2 anchors.
            (b) Rookie Year-1 shift: tier-2 rookie anchors move from
                deal-year-2 to deal-year-1, with cap_pct back-calculated
                via the 5% rookie raise (or from salaries.csv if available).

    Returns (inter_idx, needed_idx, tier, anchor_val):
        inter_idx:  list of lists — df_full positions for intermediate seasons
        needed_idx: sorted unique df_full positions that need prediction
        tier:       int array (0=fallback, 1=market anchor, 2=rookie anchor)
        anchor_val: float array of anchor cap_pct values
    """
    rs = _load_rookie_scale_set()
    event_frame = df_eval if market_events is None else market_events
    if "floor_pct" not in event_frame.columns:
        raise ValueError("market event frame must contain floor_pct")
    season_floor = event_frame.groupby("season")["floor_pct"].median().to_dict()
    floor_min = float(event_frame["floor_pct"].min())

    events: dict[str, dict[int, tuple[float, bool]]] = {}
    if extra_events:
        for p, p_events in extra_events.items():
            for s, val in p_events.items():
                events.setdefault(p, {})[int(s)] = val
    for p, s, c in zip(event_frame["player_name_norm"],
                       event_frame["season"].astype(int),
                       event_frame[TARGET].values):
        events.setdefault(p, {})[int(s)] = (float(c), False)

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

    if reprice_events:
        for p, p_events in reprice_events.items():
            for s, (_, kind, value) in p_events.items():
                if kind == "zero":
                    val = 0.0
                elif kind == "price":
                    # a mid-season total is often prorated; the annual price
                    # of a standard contract cannot sit below the minimum
                    val = max(value, season_floor.get(int(s), floor_min))
                else:
                    val = season_floor.get(int(s), floor_min)
                events.setdefault(p, {})[int(s)] = (val, True, kind == "zero")

    rookie_seasons: dict[str, list[int]] = {}
    for (p, s) in rs:
        if (p, s) in full_cp:
            rookie_seasons.setdefault(p, []).append(s)

    first_contract_cap: dict[str, dict[int, float]] = {}
    rookie_y1: dict[str, tuple[int, float]] = {}
    if expand_anchors:
        from scripts.build_external_features import norm
        debut = _load_debut_seasons()
        if debut:
            for p, s, c in zip(df_full["player_name_norm"],
                               df_full["season"].astype(int),
                               df_full[TARGET].values):
                pn = norm(str(p))
                d = debut.get(pn)
                if d is None:
                    continue
                exp = int(s) - int(d)
                if exp <= 1 and (p, int(s)) not in rs:
                    first_contract_cap.setdefault(p, {})[int(s)] = float(c)

        ROOKIE_RAISE = 1.05
        sal_lookup: dict[tuple[str, int], float] = {}
        sal_path = PROCESSED_DIR / "salaries.csv"
        if sal_path.exists():
            sal = pd.read_csv(sal_path)
            for _, r in sal.iterrows():
                cap = CAP_BY_SEASON.get(int(r["season"]))
                if cap:
                    pn = str(r["player"]).lower().strip().replace(".", "")
                    sal_lookup[(pn, int(r["season"]))] = float(r["salary"]) / cap
        for p, seasons in rookie_seasons.items():
            earliest = min(seasons)
            y1_season = earliest - 1
            cap_y1 = CAP_BY_SEASON.get(y1_season)
            if cap_y1 is None:
                continue
            pn_key = p.replace(".", "")
            if (pn_key, y1_season) in sal_lookup:
                rookie_y1[p] = (y1_season, sal_lookup[(pn_key, y1_season)])
            elif (p, earliest) in full_cp:
                cap_y2 = CAP_BY_SEASON.get(earliest)
                if cap_y2:
                    y1_salary = full_cp[(p, earliest)] * cap_y2 / ROOKIE_RAISE
                    rookie_y1[p] = (y1_season, float(y1_salary / cap_y1))

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
            if len(events[p][t0]) > 2 and events[p][t0][2]:
                tier[i] = 2
        else:
            rook = [s for s in rookie_seasons.get(p, ()) if s < T]
            if rook:
                tier[i] = 2
                if expand_anchors and p in rookie_y1:
                    dy, y1_val = rookie_y1[p]
                    if dy < T:
                        t0 = dy
                        anchor_val[i] = y1_val
                    else:
                        t0 = min(rook)
                        anchor_val[i] = full_cp[(p, t0)]
                else:
                    t0 = min(rook)
                    anchor_val[i] = full_cp[(p, t0)]
            elif expand_anchors and p in first_contract_cap:
                fc_seasons = [s for s in first_contract_cap[p] if s < T]
                if fc_seasons:
                    t0 = min(fc_seasons)
                    tier[i] = 2
                    fc_val = first_contract_cap[p][t0]
                    if fc_val < PRORATED_FLOOR:
                        anchor_val[i] = season_floor.get(t0, floor_min)
                    else:
                        anchor_val[i] = fc_val
                else:
                    inter_idx.append([])
                    continue
            else:
                inter_idx.append([])
                continue
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), tier, anchor_val


def compute_kf_column(df_eval, df_full, predict_fn, r_var,
                      players=None, extra_events=None,
                      expand_anchors=True, market_events=None,
                      reprice=True):
    """Compute kf_market_value for every row in df_eval.

    Args:
        df_eval:    target rows to compute kf for. These do not need to be
                    filtered Year-1 rows when market_events is supplied.
        df_full:    full dataset (all year_in_contract, for intermediate seasons)
        predict_fn: callable(df_subset) → predictions array.
                    Used to price intermediate seasons in df_full.
        r_var:      measurement noise variance (from base model's OOF residuals
                    or in-sample residuals)
        players:    set of player names for Q estimation (default: all in df_eval)
        extra_events: optional pre-2019 Year-1 cap_pct anchors (from
                    load_prehistory_anchors)
        expand_anchors: enable v8.14x tier-2 anchor expansions
        market_events: filtered Year-1 frame supplying historical market
                    anchors. Defaults to df_eval, preserving training-time
                    and nested-CV behaviour.
        reprice:    anchor at in-season signings (`load_reprice_events`),
                    on since v6.0.4.

    Returns:
        kf_values: array of kf_market_value, one per df_eval row.
                   NaN where no anchor exists.
    """
    inter_idx, needed_idx, tier, anchor_val = build_anchor_map(
        df_eval, df_full, extra_events=extra_events,
        expand_anchors=expand_anchors, market_events=market_events,
        reprice_events=load_reprice_events() if reprice else None)

    if players is None:
        player_frame = df_eval if market_events is None else market_events
        players = set(player_frame["player_name_norm"].unique())
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


def measurement_value(latent: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """KF measurement from the measurement model's raw output (v6.3.1).

    The output is clipped into [floor_pct, max_eligible_pct], then the
    extension cap applies where the frame records one. This is the form
    `evaluate_suite.MEASUREMENT_MODE = "clip"` scores, so the suite and the
    inference paths (`predict.py`, `export_web.py`) feed the KF the same
    measurement. A frame without `floor_pct` uses 0; a frame without the
    extension columns has no extension rows.
    """
    from src.model.stages import compose
    n = len(frame)
    lo = (frame["floor_pct"].values if "floor_pct" in frame.columns
          else np.zeros(n))
    hi = frame["max_eligible_pct"].values
    is_ext = (frame["is_extension"].fillna(False).astype(bool).values
              if "is_extension" in frame.columns else np.zeros(n, bool))
    ext_cap = (frame["ext_cap_pct"].values if "ext_cap_pct" in frame.columns
               else np.full(n, np.nan))
    return compose(np.asarray(latent, dtype=float), lo=lo, hi=hi,
                   is_extension=is_ext, ext_cap_pct=ext_cap)
