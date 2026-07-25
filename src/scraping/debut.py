"""Scrape player debut seasons from Basketball Reference player index pages.

BBRef has 26 alphabet pages (/players/a/ through /players/z/) listing
every player in history with From/To years, position, height, etc.
We parse the "From" year (data-stat="year_min") — the END-year of the
player's first NBA season — and convert to start-year convention
(debut_season = year_min - 1) to match this repo's season key.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import PROCESSED_DIR
from src.scraping.utils import fetch_html

BBREF_BASE = "https://www.basketball-reference.com"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def scrape_all_debuts() -> pd.DataFrame:
    """Scrape debut season for all NBA players from BBRef index pages."""
    rows = []
    for letter in tqdm(list(LETTERS), desc="Debut pages"):
        url = f"{BBREF_BASE}/players/{letter}/"
        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"  [error] /players/{letter}/: {e}")
            continue

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "players"})
        if not table:
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            th = tr.find("th", {"data-stat": "player"})
            if not th:
                continue
            link = th.find("a")
            player_url = link["href"] if link else None
            player_name = th.get_text(strip=True)

            ym_td = tr.find("td", {"data-stat": "year_min"})
            year_min_str = ym_td.get_text(strip=True) if ym_td else ""

            if player_url and year_min_str:
                try:
                    year_min = int(year_min_str)
                except ValueError:
                    continue
                rows.append({
                    "player_url": player_url,
                    "player_name_bbref": player_name,
                    "debut_season": year_min - 1,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = scrape_all_debuts()
    out = PROCESSED_DIR / "debut_seasons.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} players to {out}")
    print(f"Debut range: {df['debut_season'].min()} - {df['debut_season'].max()}")
