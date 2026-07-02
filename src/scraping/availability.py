"""Compute player availability from games-played data.

Availability is derived from the advanced stats already scraped (games column).
No additional scraping needed — this module computes weighted GP% over the
prior 3 seasons (weights: 0.5 / 0.3 / 0.2 per CLAUDE.md).
"""

import pandas as pd
from config import PROCESSED_DIR

# Regular season games per year (lockout-shortened seasons noted)
GAMES_IN_SEASON = {
    2018: 82,
    2019: 82,
    2020: 72,  # COVID-shortened (bubble)
    2021: 82,
    2022: 82,
    2023: 82,
    2024: 82,
    2025: 82,
}

AVAILABILITY_WEIGHTS = [0.5, 0.3, 0.2]  # most recent first


def compute_availability(advanced_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-year weighted availability for each player-season.

    Args:
        advanced_stats: DataFrame with columns [player_url, season, games]

    Returns:
        DataFrame with columns [player_url, season, availability_3yr, gp_history]
    """
    df = advanced_stats[["player_url", "season", "games"]].copy()
    df = df.dropna(subset=["player_url", "games"])
    df["max_games"] = df["season"].map(GAMES_IN_SEASON).fillna(82)
    df["gp_pct"] = df["games"] / df["max_games"]
    df = df.sort_values(["player_url", "season"])

    rows = []
    for player_url, group in df.groupby("player_url"):
        group = group.sort_values("season")
        seasons = group["season"].tolist()
        gp_pcts = group["gp_pct"].tolist()

        for i, season in enumerate(seasons):
            prior = []
            for j, w in zip(range(i, max(i - 3, -1), -1), AVAILABILITY_WEIGHTS):
                if j >= 0:
                    prior.append(gp_pcts[j] * w)

            if prior:
                total_weight = sum(
                    AVAILABILITY_WEIGHTS[k] for k in range(len(prior))
                )
                avail = sum(prior) / total_weight
            else:
                avail = gp_pcts[i] if i < len(gp_pcts) else None

            rows.append({
                "player_url": player_url,
                "season": season,
                "availability_3yr": round(avail, 4) if avail else None,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    stats_path = PROCESSED_DIR / "advanced_stats.csv"
    if not stats_path.exists():
        print(f"Run stats.py first to generate {stats_path}")
        exit(1)

    advanced = pd.read_csv(stats_path)
    avail = compute_availability(advanced)
    out = PROCESSED_DIR / "availability.csv"
    avail.to_csv(out, index=False)
    print(f"Saved {len(avail)} rows to {out}")
    print(avail.describe().to_string())
