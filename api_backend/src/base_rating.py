"""Normalize impact metrics (DARKO DPM, LEBRON, LAKER) across seasons.

Each metric is z-scored within its season so that cross-era comparisons
are apples-to-apples. The composite `base_rating` is a weighted average
of the three z-scores.
"""

import pandas as pd


METRIC_WEIGHTS = {
    "darko_dpm": 0.35,
    "lebron": 0.35,
    "laker": 0.30,
}


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
