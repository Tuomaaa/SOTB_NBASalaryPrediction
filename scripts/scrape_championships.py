"""Championship rosters by title season, from Basketball Reference.

One page per title season, not one per player: the champion's team-season page
carries both the roster and (for a title team, always) its playoff per-game
table, so 28 fetches cover every ring any player in the evaluation frame could
hold.  Everything is cached under ``data/raw/html_cache`` by
``src.scraping.utils.fetch_html``, so a re-run after a 403 resumes rather than
re-fetching.

SEASON CONVENTION
-----------------
Basketball Reference labels a season by its ENDING year -- the Raptors are the
"2019 NBA Champion" for the 2018-19 season.  This repo labels a season by its
STARTING year (``config.CAP_BY_SEASON[2025]`` is the 2025-26 cap).  The emitted
``season`` column is in REPO convention (``bbref_year - 1``); ``bbref_year`` is
carried beside it so the two are never confused.

Two membership signals are written per player, because neither alone is the
ring:

- ``on_roster``   -- appears in the team-season roster table.  Includes players
                     traded away mid-season, who did not win the ring.
- ``played_playoffs`` -- appears in that team's playoff per-game table.  Excludes
                     deep-bench players who dressed for the Finals and did not
                     play, who did win one.

``rings_thru_prev`` below counts a ring when EITHER holds and the player was
still on the roster at season's end; see ``_ring_rows``.

Usage:  python scripts/scrape_championships.py
Writes: data/processed/championships.csv
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from bs4 import BeautifulSoup, Comment

from config import PROCESSED_DIR
from src.scraping.utils import fetch_html

BASE = "https://www.basketball-reference.com"
OUT_PATH = PROCESSED_DIR / "championships.csv"
ROLLUP_PATH = PROCESSED_DIR / "rings_thru_prev.csv"

# Earliest title season to collect.  A player active in the 2019 evaluation
# frame debuted in 1998 at the earliest (Vince Carter, Dirk Nowitzki), but a
# 1998-debut player holding a 1999-2004 ring and still active in 2019 does not
# exist: the 1999-2005 title cores all retired well before the window (Spurs --
# Duncan 2016, Ginobili 2018, Parker 2018; Lakers three-peat -- Shaq 2011,
# Kobe 2016, Fisher 2014; 2004 Pistons -- Prince 2016).  The earliest ring any
# frame member actually holds is 2006 (Udonis Haslem, debut 2003, active through
# 2023 and on minimums for most of it), so 2005 is the boundary with a year of
# margin.  Narrowed from 1999 on 2026-08-29: those six seasons were pure 403
# exposure against Basketball Reference's rate limiter for zero rings.
FIRST_BBREF_YEAR = 2005
LAST_BBREF_YEAR = 2026

# Cross-check for the parsed index.  Public record; the parse is authoritative
# and any disagreement is raised rather than silently resolved.  2026 is absent
# on purpose -- it is the season this repo is pricing into and must be read off
# the page.
KNOWN_CHAMPIONS = {
    1999: "SAS", 2000: "LAL", 2001: "LAL", 2002: "LAL", 2003: "SAS",
    2004: "DET", 2005: "SAS", 2006: "MIA", 2007: "SAS", 2008: "BOS",
    2009: "LAL", 2010: "LAL", 2011: "DAL", 2012: "MIA", 2013: "MIA",
    2014: "SAS", 2015: "GSW", 2016: "CLE", 2017: "GSW", 2018: "GSW",
    2019: "TOR", 2020: "LAL", 2021: "MIL", 2022: "GSW", 2023: "DEN",
    2024: "BOS", 2025: "OKC",
}


def _norm(name: str) -> str:
    """The training frame's name convention: lowercase, accents stripped."""
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _soup(html: str) -> BeautifulSoup:
    """Parse, un-hiding the tables BBRef ships inside HTML comments."""
    soup = BeautifulSoup(html, "html.parser")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "html.parser"))
    return soup


def champions_index(retries: int = 4) -> dict[int, tuple[str, str]]:
    """``{bbref_year: (team_abbr, team_name)}`` from ``/playoffs/``.

    The index's champion cell links to the champion's team-season page, which
    is where the abbreviation comes from -- team NAMES move (Bobcats/Hornets,
    Nets) and abbreviations in the URL do not.
    """
    soup = _soup(fetch_html(f"{BASE}/playoffs/", retries=retries))
    out: dict[int, tuple[str, str]] = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        y = re.match(r"^(\d{4})", cells[0].get_text(strip=True))
        if not y:
            continue
        year = int(y.group(1))
        for cell in cells[1:]:
            a = cell.find("a", href=re.compile(r"^/teams/[A-Z]{3}/\d{4}\.html"))
            if a is None:
                continue
            m = re.match(r"^/teams/([A-Z]{3})/(\d{4})\.html", a["href"])
            if m and int(m.group(2)) == year:
                out[year] = (m.group(1), a.get_text(strip=True))
            break
    return out


def _table_players(soup: BeautifulSoup, table_id: str) -> set[tuple[str, str]]:
    """``{(player_url, display_name)}`` from one table on a team-season page."""
    t = soup.find("table", id=table_id)
    if t is None:
        return set()
    out = set()
    body = t.find("tbody") or t
    for a in body.find_all("a", href=re.compile(r"^/players/[a-z]/\w+\.html")):
        out.add((a["href"], a.get_text(strip=True)))
    return out


def champion_roster(abbr: str, bbref_year: int, retries: int = 4) -> pd.DataFrame:
    """Roster and playoff appearances for one champion's season."""
    soup = _soup(fetch_html(f"{BASE}/teams/{abbr}/{bbref_year}.html",
                            retries=retries))
    roster = _table_players(soup, "roster")
    playoffs = set()
    for tid in ("playoffs_per_game", "playoffs_totals", "playoffs_per_minute"):
        playoffs |= _table_players(soup, tid)
        if playoffs:
            break
    if not roster and not playoffs:
        raise RuntimeError(f"no roster or playoff table on {abbr}/{bbref_year}")

    rows = []
    for url, name in sorted(roster | playoffs):
        rows.append({
            "player_url": url,
            "player_name_bbref": name,
            "player_name_norm": _norm(name),
            "bbref_year": bbref_year,
            "season": bbref_year - 1,
            "team_abbr": abbr,
            "on_roster": (url, name) in roster,
            "played_playoffs": (url, name) in playoffs,
        })
    return pd.DataFrame(rows)


def rollup_rings(champs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """``rings_thru_prev`` for each (player, season) key in ``keys``.

    STRICTLY before season T: a title won in the season being priced is not an
    ex-ante fact about the contract, so ``season < T``, never ``<=``.
    """
    won = champs[champs["on_roster"] | champs["played_playoffs"]]
    by_player: dict[str, list[int]] = {}
    for name, g in won.groupby("player_name_norm"):
        by_player[name] = sorted(g["season"].unique().tolist())

    out = keys.copy()
    prev, ring_seasons = [], []
    for name, season in zip(out["player_name_norm"], out["season"]):
        seasons = [s for s in by_player.get(name, []) if s < int(season)]
        prev.append(len(seasons))
        ring_seasons.append("|".join(str(s) for s in seasons))
    out["rings_thru_prev"] = prev
    out["ring_seasons_thru_prev"] = ring_seasons
    out["has_ring_thru_prev"] = out["rings_thru_prev"] > 0
    return out


def _key_set() -> pd.DataFrame:
    """Every (player, season) the training frame prices, plus salaries.csv."""
    tr = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    tr = tr.loc[:, ~tr.columns.duplicated()]
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    sal["player_name_norm"] = sal["player"].map(_norm)
    keys = pd.concat([tr[["player_name_norm", "season"]],
                      sal[["player_name_norm", "season"]]]).astype({"season": int})
    keys = keys[(keys["season"] >= 2019) & (keys["season"] <= 2026)]
    return keys.drop_duplicates().sort_values(
        ["player_name_norm", "season"]).reset_index(drop=True)


def main() -> None:
    idx = champions_index()
    print(f"Champions index: {len(idx)} title seasons parsed "
          f"({min(idx)}-{max(idx)})")

    disagree = {y: (idx[y][0], a) for y, a in KNOWN_CHAMPIONS.items()
                if y in idx and idx[y][0] != a}
    if disagree:
        raise SystemExit(f"parsed champion disagrees with the known list: {disagree}")
    missing = [y for y in KNOWN_CHAMPIONS if y not in idx]
    if missing:
        raise SystemExit(f"index has no champion for {missing}")
    print(f"Cross-check OK against {len(KNOWN_CHAMPIONS)} known champions; "
          f"{LAST_BBREF_YEAR} parsed as {idx.get(LAST_BBREF_YEAR)}")

    frames, failed = [], []
    for year in range(FIRST_BBREF_YEAR, LAST_BBREF_YEAR + 1):
        if year not in idx:
            failed.append((year, "not in index"))
            continue
        abbr, team = idx[year]
        try:
            f = champion_roster(abbr, year)
        except Exception as exc:            # noqa: BLE001 - reported, not swallowed
            failed.append((year, f"{abbr}: {exc}"))
            print(f"  {year} {abbr}: FAILED — {exc}")
            continue
        f["team_name"] = team
        frames.append(f)
        print(f"  {year} {abbr} {team}: {len(f)} players "
              f"({int(f['played_playoffs'].sum())} played in the playoffs)")

    if failed:
        raise SystemExit(f"incomplete: {failed}. Re-run to resume from cache.")

    out = pd.concat(frames, ignore_index=True)
    cols = ["player_name_norm", "player_name_bbref", "player_url", "season",
            "bbref_year", "team_abbr", "team_name", "on_roster",
            "played_playoffs"]
    out = out[cols].sort_values(["bbref_year", "player_name_norm"])
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} — {len(out)} rows, "
          f"{out['player_name_norm'].nunique()} distinct players, "
          f"seasons {out['season'].min()}-{out['season'].max()} (repo convention)")

    rings = rollup_rings(out, _key_set())
    rings.to_csv(ROLLUP_PATH, index=False)
    print(f"Wrote {ROLLUP_PATH} — {len(rings)} rows, "
          f"{int(rings['has_ring_thru_prev'].sum())} with at least one ring")


if __name__ == "__main__":
    main()
