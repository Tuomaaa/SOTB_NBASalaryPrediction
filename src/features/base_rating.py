"""Normalize impact metrics (DARKO DPM, LEBRON, LAKER) across seasons.

Each metric is z-scored within its season so that cross-era comparisons
are apples-to-apples. The composite `base_rating` is a weighted average
of the three z-scores.
"""

import pandas as pd

from config import PROCESSED_DIR

METRIC_WEIGHTS = {
    "darko_dpm": 0.35,
    "lebron": 0.35,
    "laker": 0.30,
}

_OD_PAIRS = [
    ("darko_odpm", "darko_ddpm", "darko_od_diff"),
    ("lebron_off", "lebron_def", "lebron_od_diff"),
    ("laker_off", "laker_def", "laker_od_diff"),
]


def _zscore_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score a column within each season."""
    grouped = df.groupby("season")[col]
    return (df[col] - grouped.transform("mean")) / grouped.transform("std")


def add_base_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-normalized z-scores and composite base_rating.

    Input must have columns: season, darko_dpm, lebron, laker.
    Adds: darko_z, lebron_z, laker_z, base_rating.
    """
    df = df.copy()

    z_cols = {}
    for metric in METRIC_WEIGHTS:
        z_col = f"{metric}_z"
        if metric in df.columns:
            df[z_col] = _zscore_by_season(df, metric)
            z_cols[z_col] = METRIC_WEIGHTS[metric]

    if not z_cols:
        df["base_rating"] = None
        return df

    # Weighted average of available z-scores, handling NaN gracefully
    total_weight = 0
    df["base_rating"] = 0.0
    for z_col, weight in z_cols.items():
        mask = df[z_col].notna()
        df.loc[mask, "base_rating"] += df.loc[mask, z_col] * weight
        total_weight_series = df[list(z_cols.keys())].notna().mul(
            list(z_cols.values())
        ).sum(axis=1)

    # Normalize by actual available weight per row
    weight_per_row = pd.Series(0.0, index=df.index)
    for z_col, weight in z_cols.items():
        weight_per_row += df[z_col].notna().astype(float) * weight

    weighted_sum = pd.Series(0.0, index=df.index)
    for z_col, weight in z_cols.items():
        weighted_sum += df[z_col].fillna(0) * weight

    df["base_rating"] = weighted_sum / weight_per_row.replace(0, float("nan"))

    return df


def attach_od_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Merge O/D raw columns from impact_metrics and z-score the diffs.

    Produces darko_od_diff_z, lebron_od_diff_z, laker_od_diff_z — the
    offensive minus defensive component for each metric, z-scored within
    season. Attached at load time like kalman_filtered_stats.
    """
    od_cols = ["darko_odpm", "darko_ddpm", "lebron_off", "lebron_def",
               "laker_off", "laker_def"]
    need = [c for c in od_cols if c not in df.columns]
    if need:
        impact = pd.read_csv(
            PROCESSED_DIR / "impact_metrics.csv",
            usecols=["player_name_norm", "season"] + od_cols,
        )
        slim = impact.drop_duplicates(["player_name_norm", "season"])
        df = df.merge(slim, on=["player_name_norm", "season"], how="left",
                      suffixes=("", "_od"))

    for o_col, d_col, diff_col in _OD_PAIRS:
        if o_col in df.columns and d_col in df.columns:
            df[diff_col] = df[o_col] - df[d_col]
            df[f"{diff_col}_z"] = _zscore_by_season(df, diff_col)
    return df
