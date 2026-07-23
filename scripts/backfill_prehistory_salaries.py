"""Extract pre-2019 salaries from the already-cached BBRef player pages.

The training data starts at 2019, so the ceiling rule in
`_compute_max_eligible` (a veteran's max is at least 1.08 x his previous
season's pay) had nothing to anchor the 2019 rows on — Curry's 2019 salary sat
at 36.9% of the frozen cap with a computed ceiling of 35%. The player pages in
the HTML cache carry full salary history; the original scrape simply filtered
those seasons out. This re-parses the cache — no network — and writes
data/processed/salaries_prehistory.csv with seasons 2016-2018.

Idempotent; rerun whenever the player-page cache grows.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import CACHE_DIR, PROCESSED_DIR
from src.scraping.contracts import _find_table, _parse_salary, _season_str_to_int

SEASONS_WANTED = {2016, 2017, 2018}
OUT = PROCESSED_DIR / "salaries_prehistory.csv"


def norm(s: str) -> str:
    """lower + strip accents — matches training data's player_name_norm."""
    s = str(s).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def url_to_name_map() -> dict[str, str]:
    """player_url -> normalized display name, from the salary table."""
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    sal = sal.dropna(subset=["player_url", "player"]).drop_duplicates("player_url")
    return {u: norm(p) for u, p in zip(sal["player_url"], sal["player"])}


def main() -> None:
    name_map = url_to_name_map()
    pages = sorted(CACHE_DIR.glob("www.basketball-reference.com_players_*"))
    print(f"cached player pages: {len(pages)}")

    rows, unmatched = [], 0
    for page in pages:
        m = re.search(r"_players_(\w)_(\w+)\.html", page.name)
        if not m:
            continue
        player_url = f"/players/{m.group(1)}/{m.group(2)}.html"
        pname = name_map.get(player_url)
        if pname is None:
            unmatched += 1
            continue
        soup = BeautifulSoup(page.read_text(encoding="utf-8", errors="ignore"), "lxml")
        table = _find_table(soup, "all_salaries")
        if table is None or table.find("tbody") is None:
            continue
        for tr in table.find("tbody").find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 4:
                continue
            season_text = cells[0].get_text(strip=True)
            salary = _parse_salary(cells[3].get_text(strip=True))
            if not season_text or salary is None:
                continue
            season = _season_str_to_int(season_text)
            if season in SEASONS_WANTED:
                rows.append({"player_name_norm": pname, "player_url": player_url,
                             "season": season, "salary": salary})

    df = pd.DataFrame(rows)
    # a player traded mid-season appears once per team on some pages; keep the
    # season total by summing within (player, season)
    df = (df.groupby(["player_name_norm", "player_url", "season"], as_index=False)
            ["salary"].sum())
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(df)} rows "
          f"({df['player_name_norm'].nunique()} players), {unmatched} pages unmatched")
    print(df.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
