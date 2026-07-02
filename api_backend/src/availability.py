"""Compute player availability from games-played data.

Weighted GP% over the prior 3 seasons (weights: 0.5 / 0.3 / 0.2).
Uses games + minutes data from the impact metrics (RAPM source includes both).
"""

import pandas as pd

GAMES_IN_SEASON = {
    2018: 82, 2019: 82, 2020: 72, 2021: 82,
    2022: 82, 2023: 82, 2024: 82, 2025: 82,
}

AVAILABILITY_WEIGHTS = [0.5, 0.3, 0.2]


def compute_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-year weighted availability for each player-season.

    Input needs: player_name_norm, season, games.
    Returns the input DataFrame with `availability_3yr` added.
    """
    df = df.copy()

    if "games" not in df.columns or "player_name_norm" not in df.columns:
        df["availability_3yr"] = None
        return df

    df["max_games"] = df["season"].map(GAMES_IN_SEASON).fillna(82)
    df["gp_pct"] = df["games"] / df["max_games"]
    df = df.sort_values(["player_name_norm", "season"])

    avail_map = {}
    for player, group in df.groupby("player_name_norm"):
        group = group.sort_values("season")
        seasons = group["season"].tolist()
        gp_pcts = group["gp_pct"].tolist()

        for i, season in enumerate(seasons):
            # Current season + up to 2 prior
            prior = []
            for j in range(i, max(i - 3, -1), -1):
                if j >= 0:
                    prior.append(gp_pcts[j])

            weights = AVAILABILITY_WEIGHTS[:len(prior)]
            total_w = sum(weights)
            avail = sum(p * w for p, w in zip(prior, weights)) / total_w
            avail_map[(player, season)] = round(avail, 4)

    df["availability_3yr"] = df.apply(
        lambda r: avail_map.get((r["player_name_norm"], r["season"])), axis=1
    )
    df = df.drop(columns=["max_games", "gp_pct"])

    return df
