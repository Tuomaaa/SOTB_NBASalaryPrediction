"""Scrape per-season advanced stats from Basketball Reference.

One page per season at /leagues/NBA_{year}_advanced.html gives all players
(~750 rows) with BPM, VORP, WS, PER, USG%, TS%, and more.

For 2019-2025 that's only 7 page fetches total.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import SEASONS
from src.scraping.utils import fetch_html

BBREF_BASE = "https://www.basketball-reference.com"

KEEP_COLUMNS = {
    "name_display": "player",
    "age": "age",
    "team_name_abbr": "team",
    "pos": "position",
    "games": "games",
    "games_started": "games_started",
    "mp": "minutes",
    "per": "per",
    "ts_pct": "ts_pct",
    "usg_pct": "usg_pct",
    "ows": "ows",
    "dws": "dws",
    "ws": "ws",
    "ws_per_48": "ws_per_48",
    "obpm": "obpm",
    "dbpm": "dbpm",
    "bpm": "bpm",
    "vorp": "vorp",
}


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def scrape_season_advanced(season: int) -> pd.DataFrame:
    """Scrape advanced stats for one season. Season is the start year (e.g. 2024 = 2024-25)."""
    bbref_year = season + 1  # BBRef uses ending year in URL
    url = f"{BBREF_BASE}/leagues/NBA_{bbref_year}_advanced.html"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", {"id": "advanced"})
    if table is None:
        print(f"  [warn] No advanced table for {season}")
        return pd.DataFrame()

    thead = table.find("thead")
    header_row = thead.find_all("tr")[-1] if thead else None
    if not header_row:
        return pd.DataFrame()

    col_stats = [th.get("data-stat", "") for th in header_row.find_all("th")]

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        row = {}
        player_url = None
        for cell in cells:
            ds = cell.get("data-stat", "")
            if ds not in KEEP_COLUMNS:
                continue
            col_name = KEEP_COLUMNS[ds]
            text = cell.get_text(strip=True)

            if col_name == "player":
                row[col_name] = text
                link = cell.find("a")
                if link:
                    player_url = link.get("href", "")
            elif col_name in ("team", "position"):
                row[col_name] = text
            else:
                row[col_name] = _parse_float(text)

        if not row.get("player"):
            continue

        row["player_url"] = player_url
        row["season"] = season
        rows.append(row)

    return pd.DataFrame(rows)


def scrape_all_advanced(seasons: list[int] | None = None) -> pd.DataFrame:
    """Scrape advanced stats for all configured seasons."""
    if seasons is None:
        seasons = SEASONS
    frames = []
    for season in tqdm(seasons, desc="Advanced stats"):
        try:
            df = scrape_season_advanced(season)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  [error] Season {season}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Players traded mid-season appear multiple times (once per team + TOT).
    # Keep only the TOT (total) row for traded players.
    traded = combined[combined["team"] == "TOT"]["player"].unique()
    mask = ~((combined["player"].isin(traded)) & (combined["team"] != "TOT"))
    combined = combined[mask].reset_index(drop=True)

    return combined


if __name__ == "__main__":
    from config import PROCESSED_DIR

    df = scrape_all_advanced()
    out = PROCESSED_DIR / "advanced_stats.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Players: {df['player'].nunique()}")
    print(f"\nTop 10 BPM (2024):")
    top = df[df["season"] == 2024].nlargest(10, "bpm")[["player", "team", "bpm", "vorp", "usg_pct"]]
    print(top.to_string(index=False))
