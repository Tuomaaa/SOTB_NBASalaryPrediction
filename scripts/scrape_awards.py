"""Scrape comprehensive NBA awards + All-Star selections from Wikipedia.

Outputs data/raw/raw_external/awards_full.csv with columns:
  player_name, player_name_norm, year, award

Year = end year of NBA season (e.g. 2025 = 2024-25 season).
"""
import sys, re, time, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

from config import USER_AGENT

OUT_PATH = Path("data/raw/raw_external/awards_full.csv")


def norm(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out.replace(".", "").replace("-", " ")).strip()


def clean_player_name(raw):
    """Remove Wikipedia annotations like INJ1, REP2, NOTE1, etc."""
    return re.sub(r"(INJ\d*|REP\d*|NOTE\d*|\[.*?\]|\(.*?\))", "", raw).strip()


def fetch(url):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    if resp.status_code != 200:
        print(f"  {resp.status_code} {url}")
        return None
    return BeautifulSoup(resp.text, "html.parser")


# ─── All-Star rosters ─────────────────────────────────────────────
def scrape_allstar(year):
    """Scrape All-Star roster for a given year. Returns list of (player, award)."""
    url = f"https://en.wikipedia.org/wiki/{year}_NBA_All-Star_Game"
    soup = fetch(url)
    if not soup:
        return []

    results = []
    # Find roster tables (Pos, Player, Team, ... with Starters/Reserves sections)
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not any("Player" in h for h in headers):
            continue
        if not any("selection" in h.lower() for h in headers):
            continue

        section = None
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            text = row.get_text(strip=True)

            if "Starter" in text and len(cells) <= 2:
                section = "starter"
                continue
            elif "Reserve" in text and len(cells) <= 2:
                section = "reserve"
                continue
            elif "Head coach" in text or "coach" in text.lower():
                continue

            if section and len(cells) >= 2:
                player_raw = cells[1].get_text(strip=True)
                player = clean_player_name(player_raw)
                if player and len(player) > 2:
                    if section == "starter":
                        results.append((player, "All-Star Starter"))
                    results.append((player, "All-Star"))

    return results


# ─── Individual awards from season page ───────────────────────────
def scrape_season_awards(year):
    """Scrape MVP, DPOY, All-NBA etc. from Wikipedia season page."""
    if year <= 2003:
        fmt = f"{year-1}%E2%80%93{str(year)[2:]}_NBA_season"
    else:
        fmt = f"{year-1}%E2%80%93{str(year)[2:]}_NBA_season"
    url = f"https://en.wikipedia.org/wiki/{fmt}"
    soup = fetch(url)
    if not soup:
        return []

    results = []

    # All-NBA teams: look for tables with "First Team", "Second Team", "Third Team"
    for table in soup.find_all("table", class_="wikitable"):
        caption = table.find("caption")
        table_text = table.get_text()

        if "All-NBA" in table_text or (caption and "All-NBA" in caption.get_text()):
            team_label = None
            for row in table.find_all("tr"):
                text = row.get_text(strip=True)
                if "First" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-NBA 1st Team"
                    continue
                elif "Second" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-NBA 2nd Team"
                    continue
                elif "Third" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-NBA 3rd Team"
                    continue

                if team_label:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        player = clean_player_name(cells[1].get_text(strip=True))
                        if player and len(player) > 2:
                            results.append((player, team_label))

        if "All-Defensive" in table_text or (caption and "All-Defensive" in caption.get_text()):
            team_label = None
            for row in table.find_all("tr"):
                text = row.get_text(strip=True)
                if "First" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-Defensive 1st Team"
                    continue
                elif "Second" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-Defensive 2nd Team"
                    continue

                if team_label:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        player = clean_player_name(cells[1].get_text(strip=True))
                        if player and len(player) > 2:
                            results.append((player, team_label))

        if "All-Rookie" in table_text or (caption and "All-Rookie" in caption.get_text()):
            team_label = None
            for row in table.find_all("tr"):
                text = row.get_text(strip=True)
                if "First" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-Rookie 1st Team"
                    continue
                elif "Second" in text and "Team" in text and len(row.find_all("td")) <= 1:
                    team_label = "All-Rookie 2nd Team"
                    continue

                if team_label:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        player = clean_player_name(cells[1].get_text(strip=True))
                        if player and len(player) > 2:
                            results.append((player, team_label))

    return results


# ─── All-NBA / All-Defensive / All-Rookie teams ──────────────────
TEAM_AWARD_PAGES = {
    "All-NBA": "https://en.wikipedia.org/wiki/All-NBA_Team",
    "All-Defensive": "https://en.wikipedia.org/wiki/NBA_All-Defensive_Team",
    "All-Rookie": "https://en.wikipedia.org/wiki/NBA_All-Rookie_Team",
}


def scrape_team_awards(prefix, url, min_year=2015):
    """Parse All-NBA / All-Defensive / All-Rookie teams from Wikipedia table."""
    soup = fetch(url)
    if not soup:
        return []

    results = []
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Season" not in headers:
            continue
        has_teams = any("First" in h or "Second" in h for h in headers)
        if not has_teams:
            continue

        current_season = None
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            offset = 0
            first_text = cells[0].get_text(strip=True)
            m = re.search(r"(\d{4})[–-](\d{2,4})", first_text)
            if m:
                current_season = int(m.group(1)) + 1 if len(m.group(2)) == 2 else int(m.group(2))
                offset = 1

            if not current_season or current_season < min_year:
                continue

            remaining = [c.get_text(strip=True) for c in cells[offset:]]
            if remaining and remaining[0] in ("Guard", "Center", "Forward"):
                remaining = remaining[1:]

            teams_map = {0: "1st", 2: "2nd", 4: "3rd"}
            for start_idx, team_label in teams_map.items():
                if start_idx < len(remaining):
                    player = clean_player_name(remaining[start_idx])
                    player = re.sub(r"(\*|\^|\†)", "", player).strip()
                    if player and len(player) > 2 and not re.match(r"^\d", player):
                        if not any(x in player.lower() for x in ["nba", "team", "season"]):
                            results.append((player, f"{prefix} {team_label} Team", current_season))

    return results


# ─── Individual award winners ─────────────────────────────────────
AWARD_PAGES = {
    "MVP": "NBA_Most_Valuable_Player_Award",
    "Finals MVP": "NBA_Finals_Most_Valuable_Player_Award",
    "Defensive Player of the Year": "NBA_Defensive_Player_of_the_Year_Award",
    "Most Improved Player": "NBA_Most_Improved_Player_Award",
    "Sixth Man of the Year": "NBA_Sixth_Man_of_the_Year_Award",
    "Rookie of the Year": "NBA_Rookie_of_the_Year_Award",
    "Clutch Player of the Year": "NBA_Clutch_Player_of_the_Year_Award",
}


def scrape_award_winners(award_name, wiki_page, min_year=2014):
    """Scrape award winner list from Wikipedia."""
    url = f"https://en.wikipedia.org/wiki/{wiki_page}"
    soup = fetch(url)
    if not soup:
        return []

    results = []
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # Find tables with season/year + player columns
        has_season = any("season" in h or "year" in h for h in headers)
        has_player = any("player" in h or "winner" in h for h in headers)
        if not (has_season and has_player):
            continue

        season_idx = next((i for i, h in enumerate(headers) if "season" in h or "year" in h), None)
        player_idx = next((i for i, h in enumerate(headers) if "player" in h or "winner" in h), None)
        if season_idx is None or player_idx is None:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(season_idx, player_idx):
                continue
            season_text = cells[season_idx].get_text(strip=True)
            player_text = clean_player_name(cells[player_idx].get_text(strip=True))

            # Parse year from season text (e.g., "2024–25" -> 2025)
            m = re.search(r"(\d{4})[–-](\d{2,4})", season_text)
            if m:
                year = int(m.group(1)) + 1 if len(m.group(2)) == 2 else int(m.group(2))
            else:
                m2 = re.search(r"(\d{4})", season_text)
                if m2:
                    year = int(m2.group(1))
                else:
                    continue

            if year >= min_year and player_text and len(player_text) > 2:
                results.append((player_text, award_name, year))

    return results


def main():
    all_awards = []

    # 1. Individual award winners
    print("Scraping individual award winners...")
    for award_name, wiki_page in AWARD_PAGES.items():
        entries = scrape_award_winners(award_name, wiki_page)
        for player, award, year in entries:
            all_awards.append({"player_name": player, "year": year, "award": award})
        print(f"  {award_name}: {len(entries)} entries")
        time.sleep(1)

    # 2. All-Star rosters (2015-2026)
    print("\nScraping All-Star rosters...")
    for year in range(2015, 2027):
        entries = scrape_allstar(year)
        for player, award in entries:
            all_awards.append({"player_name": player, "year": year, "award": award})
        n_stars = sum(1 for _, a in entries if a == "All-Star")
        n_starters = sum(1 for _, a in entries if a == "All-Star Starter")
        print(f"  {year}: {n_stars} all-stars, {n_starters} starters")
        time.sleep(1)

    # 3. All-NBA / All-Defensive / All-Rookie teams
    print("\nScraping team awards...")
    for prefix, url in TEAM_AWARD_PAGES.items():
        entries = scrape_team_awards(prefix, url)
        for player, award, year in entries:
            all_awards.append({"player_name": player, "year": year, "award": award})
        print(f"  {prefix}: {len(entries)} entries ({min(e[2] for e in entries) if entries else '?'}-{max(e[2] for e in entries) if entries else '?'})")
        time.sleep(1)

    # Build DataFrame
    df = pd.DataFrame(all_awards)
    df["player_name_norm"] = df["player_name"].apply(norm)
    df = df.drop_duplicates(["player_name_norm", "year", "award"])

    print(f"\n{'='*60}")
    print(f"Total: {len(df)} award entries")
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"\nAward distribution:")
    print(df["award"].value_counts().to_string())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
