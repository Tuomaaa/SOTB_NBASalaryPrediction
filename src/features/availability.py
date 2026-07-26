"""Compute player availability from games-played data.

Weighted GP% over the prior 3 seasons (weights: 0.5 / 0.3 / 0.2).
Uses games + minutes data from the impact metrics (RAPM source includes both).
"""

import pandas as pd

GAMES_IN_SEASON = {
    2018: 82, 2019: 82, 2020: 72, 2021: 72,
    2022: 82, 2023: 82, 2024: 82, 2025: 82, 2026: 82,
}

AVAILABILITY_WEIGHTS = [0.5, 0.3, 0.2]


def compute_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-year weighted availability for each player-season.

    Input needs: player_name_norm, season, games.
    Missing source seasons are skipped and the weights are renormalized over
    observed seasons. An observed zero remains a zero. The accompanying
    ``availability_3yr_coverage`` records the share of the nominal 1.0 weight
    that was observed, so missingness is auditable without becoming a model
    feature.
    """
    df = df.copy()

    if "games" not in df.columns or "player_name_norm" not in df.columns:
        df["availability_3yr"] = None
        return df

    df["max_games"] = df["season"].map(GAMES_IN_SEASON).fillna(82)
    df["gp_pct"] = df["games"] / df["max_games"]
    df = df.sort_values(["player_name_norm", "season"])

    avail_map = {}
    coverage_map = {}
    for player, group in df.groupby("player_name_norm"):
        group = group.sort_values("season")
        gp_by_season = dict(zip(group["season"].astype(int), group["gp_pct"]))

        for season in group["season"].astype(int):
            observed = [
                (gp_by_season.get(season - lag), weight)
                for lag, weight in enumerate(AVAILABILITY_WEIGHTS)
                if pd.notna(gp_by_season.get(season - lag))
            ]
            observed_weight = sum(weight for _, weight in observed)
            avail_map[(player, season)] = (
                round(sum(value * weight for value, weight in observed)
                      / observed_weight, 4)
                if observed_weight else None
            )
            coverage_map[(player, season)] = round(observed_weight, 2)

    df["availability_3yr"] = df.apply(
        lambda r: avail_map.get((r["player_name_norm"], r["season"])), axis=1
    )
    df["availability_3yr_coverage"] = df.apply(
        lambda r: coverage_map.get((r["player_name_norm"], r["season"])), axis=1
    )
    df = df.drop(columns=["max_games", "gp_pct"])

    return df
