"""Scrape player height from Basketball Reference player index pages.

BBRef has 26 alphabet pages (/players/a/ through /players/z/) listing
every player in history with height, weight, position, etc.
We parse height (e.g. "6-11") into inches (83) and match by player_url.
"""

import re
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


def _height_to_inches(ht_str: str) -> int | None:
    m = re.match(r"(\d+)-(\d+)", ht_str.strip())
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


def scrape_all_heights() -> pd.DataFrame:
    """Scrape height for all NBA players from BBRef index pages."""
    rows = []
    for letter in tqdm(list(LETTERS), desc="Height pages"):
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

            ht_td = tr.find("td", {"data-stat": "height"})
            ht_str = ht_td.get_text(strip=True) if ht_td else ""
            height = _height_to_inches(ht_str) if ht_str else None

            if player_url and height:
                rows.append({
                    "player_url": player_url,
                    "player_name_bbref": player_name,
                    "height_inches": height,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = scrape_all_heights()
    out = PROCESSED_DIR / "heights.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} players to {out}")
    print(f"Height range: {df['height_inches'].min()} - {df['height_inches'].max()} inches")
