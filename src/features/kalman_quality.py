"""Kalman-filtered player quality estimate (`kf_q`).

State model:
    x_{t+1} = x_t + drift(age) + w,   w ~ N(0, Q)
    z_t     = [darko_z, lebron_z, rapm_z] = x_t + v,   v ~ N(0, diag(R))

All parameters estimated from metric data only (never the target):
    R from pairwise metric disagreement (3 equations, 3 unknowns)
    drift(age) from mean year-over-year change in row-mean z, bucketed by age
    Q from residual yoy variance after drift removal minus measurement noise

The filter runs causally per player (sorted by season): row (p, s) sees only
seasons <= s.  The metrics in each row are already lagged (season s-1
performance), matching the project's intentional-lag convention.

Attached at load time (train.py::load_training_data) rather than baked into
training_data_v2.csv, the same way attach_waiver_interactions and
attach_playoff_mpg are: the column is a pure function of columns already in the
table, so a rebuild would move unrelated columns for nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

_METRICS = ["darko_dpm_z", "lebron_z", "rapm_z"]


def _estimate_params(td: pd.DataFrame) -> tuple[np.ndarray, dict, float]:
    """R, drift-by-age-bucket, Q — all from metric trajectories."""
    d, l, r = (td[m] for m in _METRICS)
    v_dl, v_dr, v_lr = (d - l).var(), (d - r).var(), (l - r).var()
    R = np.clip(np.array([
        (v_dl + v_dr - v_lr) / 2,
        (v_dl + v_lr - v_dr) / 2,
        (v_dr + v_lr - v_dl) / 2,
    ]), 0.02, None)

    td = td.sort_values(["player_name_norm", "season"])
    td["_row_mean"] = td[_METRICS].mean(axis=1)
    g = td.groupby("player_name_norm")
    td["_yoy"] = g["_row_mean"].diff()
    td["_gap"] = g["season"].diff()
    td["_prev_age"] = g["age"].shift()
    consec = td[td["_gap"] == 1].dropna(subset=["_yoy", "_prev_age"]).copy()

    bins = [0, 22, 25, 28, 31, 34, 50]
    consec["_age_bin"] = pd.cut(consec["_prev_age"], bins=bins)
    drift_by_bin = consec.groupby("_age_bin", observed=True)["_yoy"].mean()
    drift_map = {b: float(v) for b, v in drift_by_bin.items()}

    noise_rowmean = R.mean() / 3
    resid_var = float((consec["_yoy"] - consec.groupby("_age_bin", observed=True)
                       ["_yoy"].transform("mean")).var())
    Q = max(resid_var - 2 * noise_rowmean, 0.01)

    return R, drift_map, Q


def _drift_lookup(age: float, drift_map: dict, bins=None) -> float:
    if bins is None:
        bins = [0, 22, 25, 28, 31, 34, 50]
    idx = pd.cut(pd.Series([age]), bins=bins)[0]
    return drift_map.get(idx, 0.0)


def _run_filter(td: pd.DataFrame, R: np.ndarray,
                drift_map: dict, Q: float) -> pd.DataFrame:
    """Causal per-player scalar Kalman filter with sequential updates."""
    td = td.sort_values(["player_name_norm", "season"])
    out = []
    for player, grp in td.groupby("player_name_norm"):
        x, P = 0.0, 1.0
        prev_season = None
        for _, row in grp.iterrows():
            s = int(row["season"])
            steps = 1 if prev_season is None else max(int(s - prev_season), 1)
            age0 = row["age"] - steps
            for i in range(steps):
                x = x + _drift_lookup(age0 + i, drift_map)
                P = P + Q
            z = np.array([row[m] for m in _METRICS], dtype=float)
            avail = ~np.isnan(z)
            if avail.any():
                for zi, ri in zip(z[avail], R[avail]):
                    K = P / (P + ri)
                    x = x + K * (zi - x)
                    P = (1 - K) * P
            out.append({"player_name_norm": player, "season": s, "kalman_filtered_stats": x})
            prev_season = s
    return pd.DataFrame(out)


def attach_kalman_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate KF parameters from trajectories and merge kf_q onto df."""
    td = pd.read_csv(
        PROCESSED_DIR / "training_data_v2.csv",
        usecols=["player_name_norm", "season", "age"] + _METRICS,
    )
    R, drift_map, Q = _estimate_params(td.dropna(subset=["age"]))
    kf = _run_filter(td, R, drift_map, Q)
    df = df.merge(kf, on=["player_name_norm", "season"], how="left")
    df["kalman_filtered_stats"] = df["kalman_filtered_stats"].fillna(
        df["kalman_filtered_stats"].median())
    return df
