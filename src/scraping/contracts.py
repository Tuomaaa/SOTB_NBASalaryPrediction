"""Scrape NBA contract/salary data from Basketball Reference.

Two data paths:
  1. Team contract pages (/contracts/TEAM.html) — current + future salaries
     for every rostered player. 30 page fetches cover the whole league.
  2. Player salary pages (/players/x/name.html) — full salary history back
     to rookie year. Needed for historical seasons.

All raw HTML is cached in data/raw/html_cache/ to avoid re-scraping.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm

from config import CAP_BY_SEASON, PROCESSED_DIR, RAW_DIR, SEASONS
from src.scraping.utils import fetch_html

BBREF_BASE = "https://www.basketball-reference.com"

TEAM_ABBREVS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# Historical abbreviation changes on BBRef
ABBREV_ALIASES = {
    "NJN": "BRK",
    "CHA": "CHO",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
    "WSB": "WAS",
}


def _parse_salary(text: str) -> int | None:
    """Parse '$1,234,567' into 1234567. Returns None for empty/non-salary."""
    text = text.strip()
    if not text or text == "-":
        return None
    cleaned = re.sub(r"[,$\s]", "", text)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _season_str_to_int(season_str: str) -> int:
    """Convert '2023-24' to 2023 (the start year, matching CAP_BY_SEASON keys)."""
    parts = season_str.strip().split("-")
    if len(parts) == 2:
        return int(parts[0])
    return int(season_str)


def _find_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """Find a table by id, checking both direct DOM and HTML comments.

    BBRef hides some tables in <!-- comments --> that get unpacked by JS.
    """
    table = soup.find("table", {"id": table_id})
    if table:
        return table
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in str(comment):
            comment_soup = BeautifulSoup(str(comment), "lxml")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                return table
    return None


# ---------------------------------------------------------------------------
# Path 1: team contract pages (current + future salaries)
# ---------------------------------------------------------------------------

def scrape_team_contracts(team: str) -> pd.DataFrame:
    """Scrape a single team's contract page. Returns one row per player-season."""
    url = f"{BBREF_BASE}/contracts/{team}.html"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    table = _find_table(soup, "contracts")
    if table is None:
        print(f"  [warn] No contracts table for {team}")
        return pd.DataFrame()

    thead = table.find("thead")
    header_ths = thead.find_all("tr")[-1].find_all("th") if thead else []
    columns = [th.get_text(strip=True) for th in header_ths]

    season_cols = [c for c in columns if re.match(r"\d{4}-\d{2}", c)]

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 3:
            continue
        player_cell = cells[0]
        player_name = player_cell.get_text(strip=True)
        if not player_name or player_name.startswith("Totals"):
            continue

        player_link = player_cell.find("a")
        player_url = player_link["href"] if player_link else ""

        age_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        age = int(age_text) if age_text.isdigit() else None

        for i, season_label in enumerate(season_cols):
            col_idx = columns.index(season_label)
            if col_idx < len(cells):
                salary = _parse_salary(cells[col_idx].get_text(strip=True))
                if salary and salary > 0:
                    season_end = _season_str_to_int(season_label)
                    rows.append({
                        "player": player_name,
                        "player_url": player_url,
                        "team": team,
                        "season": season_end,
                        "salary": salary,
                        "age": age,
                        "source": "bbref_team_contracts",
                    })

    return pd.DataFrame(rows)


def scrape_all_team_contracts() -> pd.DataFrame:
    """Scrape contract pages for all 30 teams. Returns combined DataFrame."""
    frames = []
    for team in tqdm(TEAM_ABBREVS, desc="Team contracts"):
        try:
            df = scrape_team_contracts(team)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  [error] {team}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Path 2: player salary history pages (all historical seasons)
# ---------------------------------------------------------------------------

def scrape_player_salaries(player_url: str) -> pd.DataFrame:
    """Scrape salary history from a player's BBRef page.

    Args:
        player_url: relative URL like '/players/t/tatumja01.html'

    Returns one row per season the player was paid.
    """
    url = f"{BBREF_BASE}{player_url}"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    table = _find_table(soup, "all_salaries")
    if table is None:
        return pd.DataFrame()

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 4:
            continue
        season_text = cells[0].get_text(strip=True)
        team_text = cells[1].get_text(strip=True)
        salary = _parse_salary(cells[3].get_text(strip=True))

        if not season_text or salary is None:
            continue

        season = _season_str_to_int(season_text)
        rows.append({
            "player_url": player_url,
            "team_name": team_text,
            "season": season,
            "salary": salary,
            "source": "bbref_player_salary",
        })

    return pd.DataFrame(rows)


def scrape_historical_salaries(player_urls: list[str]) -> pd.DataFrame:
    """Scrape salary history for a list of players.

    Args:
        player_urls: list of relative BBRef player URLs

    Filters to SEASONS defined in config.
    """
    frames = []
    for url in tqdm(player_urls, desc="Player salaries"):
        try:
            df = scrape_player_salaries(url)
            if not df.empty:
                df = df[df["season"].isin(SEASONS)]
                frames.append(df)
        except Exception as e:
            print(f"  [error] {url}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def _add_cap_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add cap_pct column = salary / salary_cap for that season."""
    df = df.copy()
    df["salary_cap"] = df["season"].map(CAP_BY_SEASON)
    df["cap_pct"] = df["salary"] / df["salary_cap"]
    return df


def build_salary_dataset(include_history: bool = False) -> pd.DataFrame:
    """Main entry point: scrape salaries and produce a clean DataFrame.

    Args:
        include_history: if True, also scrape individual player pages for
            seasons not covered by current team contract pages. Much slower
            (500+ page fetches).
    """
    print("=== Scraping current team contracts ===")
    current = scrape_all_team_contracts()
    print(f"  Got {len(current)} player-season rows from team pages")

    if include_history and not current.empty:
        player_urls = current["player_url"].dropna().unique().tolist()
        player_urls = [u for u in player_urls if u]
        print(f"\n=== Scraping salary history for {len(player_urls)} players ===")
        history = scrape_historical_salaries(player_urls)
        print(f"  Got {len(history)} historical rows")

        if not history.empty:
            history["player"] = history["player_url"].str.extract(
                r"/players/\w/(\w+)\.html"
            )
            combined = pd.concat([current, history], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["player_url", "season"], keep="first"
            )
        else:
            combined = current
    else:
        combined = current

    combined = _add_cap_pct(combined)

    out_path = PROCESSED_DIR / "salaries.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    return combined


if __name__ == "__main__":
    df = build_salary_dataset(include_history=False)
    print(df.head(10))
    print(f"\nSeasons covered: {sorted(df['season'].unique())}")
    print(f"Teams: {df['team'].nunique()}")
    print(f"Players: {df['player'].nunique()}")
