"""Five-arm feature batch: column builders + a persisted column manifest.

Produced by the 2026-07-25 feature-batch dispatch. Each arm's columns are
delivered here whether or not the arm cleared its regression gate — the route
classifier (2026-07-25-route-mixture) consumes them regardless. See
docs/briefs/2026-07-25-feature-batch.RESULT.md for the paired evidence.

    python scripts/feature_batch.py     # writes data/processed/feature_batch_columns.csv

Every builder takes the evaluation frame (as produced by
src.model.evaluate_suite.load_evaluation_frame) and returns new columns indexed
like the frame. The builders are the SHIP FORM: the arm-c prev_cap_pct here is
byte-for-byte the column scripts.phase3.build_contract_features(prev_mode=
"prev-season") writes on the full frame.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, PROCESSED_DIR, RAW_DIR

IMPACT = ["darko_dpm", "lebron", "laker"]
PROD_FEATS = ["darko_dpm_z", "lebron_z", "laker_z", "mpg", "usage_pct", "ast_pct"]


# ─── shared source tables ────────────────────────────────────────────
def _full_training() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    return df.loc[:, ~df.columns.duplicated()]


def _impact_z() -> pd.DataFrame:
    """Within-season z-scores of the three impact metrics over the impact_metrics
    population (all rated players), so arm-(a) lags are internally consistent."""
    im = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")
    out = im[["player_name_norm", "season"]].copy()
    for m in IMPACT:
        g = im.groupby("season")[m]
        out[f"{m}_z"] = (im[m] - g.transform("mean")) / g.transform("std")
    return out


# ─── Arm (a) — trend features ────────────────────────────────────────
def build_trend(df: pd.DataFrame) -> pd.DataFrame:
    """1-yr delta, 3-yr slope, peak-minus-current per impact metric (z), native
    NaN where the lag is missing, plus a `trend_has_prev` coverage indicator."""
    z = _impact_z()
    zmap = {m: dict(zip(zip(z["player_name_norm"], z["season"]), z[f"{m}_z"]))
            for m in IMPACT}
    hist = {m: {} for m in IMPACT}
    for m in IMPACT:
        for (p, s), v in zmap[m].items():
            hist[m].setdefault(p, {})[int(s)] = v
    cols = {}
    for m in IMPACT:
        d1, slope3, peakd = [], [], []
        for p, s in zip(df["player_name_norm"], df["season"].astype(int)):
            zt = zmap[m].get((p, s), np.nan)
            zp = zmap[m].get((p, s - 1), np.nan)
            d1.append(zt - zp if (zt == zt and zp == zp) else np.nan)
            pts = [(k, hist[m].get(p, {}).get(s + k)) for k in (-2, -1, 0)]
            pts = [(k, v) for k, v in pts if v is not None and v == v]
            slope3.append(float(np.polyfit([k for k, _ in pts], [v for _, v in pts], 1)[0])
                          if len(pts) >= 2 else np.nan)
            prior = [v for k, v in hist[m].get(p, {}).items() if k <= s and v == v]
            peakd.append(max(prior) - zt if (prior and zt == zt) else np.nan)
        cols[f"{m}_z_d1"] = d1
        cols[f"{m}_z_slope3"] = slope3
        cols[f"{m}_z_peakd"] = peakd
    out = pd.DataFrame(cols, index=df.index)
    out["trend_has_prev"] = [1.0 if (p, s - 1) in zmap["darko_dpm"] else 0.0
                             for p, s in zip(df["player_name_norm"], df["season"].astype(int))]
    return out


# ─── Arm (b) — stats as of signing ───────────────────────────────────
def _covering_extension_asof(df: pd.DataFrame):
    """(is_priced_early, as_of_season) per row: a row is priced-early when its
    covering contract is an extension signed in an earlier season; the market
    saw production through signing_season - 1."""
    from scripts.parse_signing_dates import contract_spans, covering_contract
    spans = contract_spans()
    sal = (df["cap_pct"] * df["season"].map(CAP_BY_SEASON)).values
    early, asof = [], []
    for p, s, d in zip(df["player_name_norm"], df["season"].astype(int), sal):
        cov = covering_contract(spans, p, int(s), salary=float(d))
        if cov is None or int(cov["is_extension"]) != 1 \
                or int(cov["signing_season"]) >= int(s):
            early.append(0.0); asof.append(np.nan); continue
        early.append(1.0); asof.append(int(cov["signing_season"]) - 1)
    return np.array(early), np.array(asof, float)


def build_asof_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Delta form (the shippable arm-b column set): (current - as_of) per
    production feature + an is_priced_early indicator; 0 where not applicable."""
    early, asof = _covering_extension_asof(df)
    full = _full_training()
    lut = {f: dict(zip(zip(full["player_name_norm"], full["season"]), full[f]))
           for f in PROD_FEATS}
    cols = {f"{f}_signdelta": np.zeros(len(df)) for f in PROD_FEATS}
    for i in np.where(early == 1)[0]:
        p = df["player_name_norm"].iloc[i]; a = int(asof[i])
        for f in PROD_FEATS:
            va = lut[f].get((p, a), np.nan); vc = df[f].iloc[i]
            if va == va and vc == vc:
                cols[f"{f}_signdelta"][i] = vc - va
    out = pd.DataFrame(cols, index=df.index)
    out["is_priced_early"] = early
    return out


def asof_replace(df: pd.DataFrame) -> pd.DataFrame:
    """Replace form (destructive; provided for the classifier, not shipped):
    production features swapped for their as-of-signing values on priced-early
    extension rows. Returns a full PROD_FEATS frame."""
    early, asof = _covering_extension_asof(df)
    full = _full_training()
    lut = {f: dict(zip(zip(full["player_name_norm"], full["season"]), full[f]))
           for f in PROD_FEATS}
    out = df[PROD_FEATS].copy()
    for i in np.where(early == 1)[0]:
        p = df["player_name_norm"].iloc[i]; a = int(asof[i])
        for f in PROD_FEATS:
            v = lut[f].get((p, a), np.nan)
            if v == v:
                out.iloc[i, out.columns.get_loc(f)] = v
    return out.rename(columns={f: f"{f}_asof" for f in PROD_FEATS})


# ─── Arm (c) — prev_cap_pct = previous season's actual pay (ship form) ─
def build_prev_cap_uniform(df: pd.DataFrame) -> pd.DataFrame:
    """Arm B semantics. Runs the SHIP builder on the full frame and maps the
    column onto df, so it matches phase3.build_contract_features(prev_mode=
    "prev-season") byte-for-byte (the eval frame lacks the rookie-scale rows
    that calibrate the fill map, so it cannot be rebuilt on df alone)."""
    from scripts.phase3 import build_contract_features
    full = _full_training().copy()
    full["cap_pct"] = full["salary"] / full["season"].map(CAP_BY_SEASON)
    built = build_contract_features(full, prev_mode="prev-season")
    ship = dict(zip(zip(built["player_name_norm"], built["season"]), built["prev_cap_pct"]))
    return pd.DataFrame(
        {"prev_cap_pct_prevseason": [ship.get((p, int(s)), np.nan)
                                     for p, s in zip(df["player_name_norm"], df["season"])]},
        index=df.index)


# ─── Arm (d) — prior-year estimated value ────────────────────────────
POINTS_PER_WIN = 30.0
POSS_PER_MIN = 2.0
N_TEAMS = 30
TOTAL_WINS_SEASON = 30 * 82 / 2  # 1230


def _est_cap_share(darko, minutes):
    """Prior-year est value as a share of the season cap. $/win is a per-season
    constant from the cap (N_TEAMS*cap/TOTAL_WINS); it cancels against the cap,
    so the share is era-neutral: est_share = wins * N_TEAMS / TOTAL_WINS."""
    if not (darko == darko and minutes == minutes):
        return np.nan
    wins = (darko / 100.0) * (minutes * POSS_PER_MIN) / POINTS_PER_WIN
    return wins * N_TEAMS / TOTAL_WINS_SEASON


def build_est_value(df: pd.DataFrame) -> pd.DataFrame:
    """d1: est_value_prev (native NaN, no prior year) + est_has_prev coverage."""
    full = _full_training()
    dk = dict(zip(zip(full["player_name_norm"], full["season"]), full["darko_dpm"]))
    mn = dict(zip(zip(full["player_name_norm"], full["season"]), full["mpg"] * full["games"]))
    est = [_est_cap_share(dk.get((p, s - 1), np.nan), mn.get((p, s - 1), np.nan))
           for p, s in zip(df["player_name_norm"], df["season"].astype(int))]
    return pd.DataFrame({"est_value_prev": est,
                         "est_has_prev": [1.0 if e == e else 0.0 for e in est]},
                        index=df.index)


def build_est_value_d2(df: pd.DataFrame) -> pd.DataFrame:
    """d2: rookie-exit rows only, prev_cap_pct replaced in-slot by the prior-year
    est value clipped to [floor_pct, max_eligible_pct]. Needs floor_pct and
    max_eligible_pct on df (present in the evaluation frame)."""
    from src.model.train import _load_rookie_scale_set
    full = _full_training()
    dk = dict(zip(zip(full["player_name_norm"], full["season"]), full["darko_dpm"]))
    mn = dict(zip(zip(full["player_name_norm"], full["season"]), full["mpg"] * full["games"]))
    rs = _load_rookie_scale_set()
    new = df["prev_cap_pct"].astype(float).copy()
    for i, (p, s) in enumerate(zip(df["player_name_norm"], df["season"].astype(int))):
        if (p, s - 1) not in rs:
            continue
        e = _est_cap_share(dk.get((p, s - 1), np.nan), mn.get((p, s - 1), np.nan))
        if e == e:
            new.iloc[i] = float(np.clip(e, df["floor_pct"].iloc[i], df["max_eligible_pct"].iloc[i]))
    return pd.DataFrame({"prev_cap_pct_d2": new}, index=df.index)


# ─── Arm (e) — rookie-award tier + name-cleaned award score ───────────
def _clean_award_name(s: str) -> str:
    """Strip footnote junk (^ § † digits, replacement char) that awards_full's
    player_name_norm carries and scripts.build_external_features.norm does not
    remove — see ISSUES entry on the award name join."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z .'-]", "", str(s))).strip()


def _awards_clean() -> pd.DataFrame:
    aw = pd.read_csv(RAW_DIR / "raw_external" / "awards_full.csv")
    aw["player_name_norm"] = aw["player_name_norm"].map(_clean_award_name)
    return aw


def build_rookie_award_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player ordinal rookie honor (ROY=3, All-Rookie 1st=2, 2nd=1, none=0),
    names cleaned so ROY winners (all footnote-marked) match."""
    aw = _awards_clean()
    tier = {"Rookie of the Year": 3, "All-Rookie 1st Team": 2, "All-Rookie 2nd Team": 1}
    aw["t"] = aw["award"].map(tier).fillna(0)
    best = aw.groupby("player_name_norm")["t"].max().to_dict()
    return pd.DataFrame({"rookie_award_tier": [best.get(p, 0.0) for p in df["player_name_norm"]]},
                        index=df.index)


def _award_score_col(df, W, name):
    from scripts.build_external_features import _step_decay_factor
    aw = _awards_clean()
    aw = aw[aw["year"] >= 2017].copy()
    aw["award_pts"] = aw["award"].map(W).fillna(0)
    aw = aw[aw["award_pts"] > 0]
    yearly = aw.groupby(["player_name_norm", "year"])["award_pts"].sum().reset_index()
    hist = {p: list(zip(g["year"].astype(int), g["award_pts"]))
            for p, g in yearly.groupby("player_name_norm")}
    score = []
    for p, s in zip(df["player_name_norm"], df["season"].astype(int)):
        score.append(round(sum(pts * _step_decay_factor((s - 1) - yr)
                               for yr, pts in hist.get(p, [])
                               if _step_decay_factor((s - 1) - yr) > 0), 3))
    return pd.DataFrame({name: score}, index=df.index)


def build_award_variants(df: pd.DataFrame) -> pd.DataFrame:
    """award_score_cum with cleaned names, shipped weights (clean-only control)
    and with heavier rookie weights (reweight)."""
    from scripts.build_external_features import AWARD_WEIGHTS
    clean = _award_score_col(df, dict(AWARD_WEIGHTS), "award_score_cum_clean")
    W = dict(AWARD_WEIGHTS)
    W["Rookie of the Year"] = 4.0
    W["All-Rookie 1st Team"] = 3.0
    W["All-Rookie 2nd Team"] = 1.5
    rew = _award_score_col(df, W, "award_score_cum_reweight")
    return pd.concat([clean, rew], axis=1)


# ─── manifest ────────────────────────────────────────────────────────
def build_all(df: pd.DataFrame) -> pd.DataFrame:
    key = df[["player_name_norm", "season"]].copy()
    parts = [key, build_trend(df), build_asof_delta(df), build_prev_cap_uniform(df),
             build_est_value(df), build_est_value_d2(df),
             build_rookie_award_tier(df), build_award_variants(df)]
    return pd.concat(parts, axis=1)


def main():
    from src.model.evaluate_suite import load_evaluation_frame
    df, _ = load_evaluation_frame()
    out = build_all(df)
    path = PROCESSED_DIR / "feature_batch_columns.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows, {out.shape[1]} cols)")
    print("columns:", [c for c in out.columns if c not in ("player_name_norm", "season")])


if __name__ == "__main__":
    main()
