"""Scrape 2018-19 team-season salary tables into salaries_prehistory.csv.

The FEATURE prev_cap_pct (ISSUES #13) and the ceiling anchor
(_load_prev_season_cap_pct) both need a player's 2018-19 pay to price the
147 season-2019 year-1 rows. The prehistory table so far carries only the
153 players who happened to have a cached BBRef PLAYER page
(backfill_prehistory_salaries.py). This reads the 30 BBRef TEAM-season pages
for 2018-19 (/teams/<ABBR>/2019.html, `salaries2` table), which enumerate the
whole league, and EXTENDS the table.

Discipline (per the worker brief and CLAUDE.md):
  - EXTEND-ONLY. Existing (player, season) rows are carried byte-for-byte;
    a scraped value that CONFLICTS with an existing row is reported and the
    existing row is kept — never overwritten.
  - A failed team page degrades to that team missing from the new rows, never
    to a deleted existing row. Rate-limited + cached via fetch_html.
  - season label 2018 (= 2018-19), matching the existing Klay row.

Rerun-safe: idempotent given the cache. Network only for uncached team pages.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import CAP_BY_SEASON, PROCESSED_DIR
from src.scraping.contracts import (
    BBREF_BASE, TEAM_ABBREVS, _find_table, _parse_salary,
)
from src.scraping.utils import fetch_html
from scripts.backfill_prehistory_salaries import norm

SEASON = 2018  # 2018-19
OUT = PROCESSED_DIR / "salaries_prehistory.csv"
# a scraped season-total this far (in $) from an existing prehistory value is a
# genuine conflict to surface, not scrape/rounding noise
CONFLICT_TOL = 50_000

from config import RAW_DIR as _RAW_DIR
_ALIAS_PATH = _RAW_DIR / "raw_external" / "player_name_aliases.csv"


def _load_aliases() -> dict[str, str]:
    if not _ALIAS_PATH.exists():
        return {}
    al = pd.read_csv(_ALIAS_PATH)
    return dict(zip(al["old_name"], al["current_name"]))


def scrape_team_2019_salaries(team: str) -> pd.DataFrame:
    """Parse the `salaries2` table off /teams/<team>/2019.html.

    Returns one row per player on that team's 2018-19 books:
    player_name_norm, player_url, season, salary. Empty frame if the page
    403s or the table is absent — the caller keeps going (degrade to missing).
    """
    url = f"{BBREF_BASE}/teams/{team}/2019.html"
    try:
        # extra patience: BBRef 403s a large share of a burst; more retries with
        # the built-in exponential backoff lets a rerun pass mop up stragglers.
        html = fetch_html(url, retries=4)
    except Exception as e:  # 403 after retries, timeout, etc.
        print(f"  [warn] {team}: fetch failed, team omitted ({e})")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")
    table = _find_table(soup, "salaries2")
    if table is None or table.find("tbody") is None:
        print(f"  [warn] {team}: no salaries2 table, team omitted")
        return pd.DataFrame()

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all(["th", "td"])
        cell = {c.get("data-stat"): c for c in cells}
        pcell = cell.get("player")
        scell = cell.get("salary")
        if pcell is None or scell is None:
            continue
        name = pcell.get_text(strip=True)
        if not name or name.startswith("Totals"):
            continue
        link = pcell.find("a")
        player_url = link["href"] if link else ""
        salary = _parse_salary(scell.get_text(strip=True))
        if salary is None or salary <= 0:
            continue
        rows.append({
            "player_name_norm": norm(name),
            "player_url": player_url,
            "season": SEASON,
            "salary": salary,
            "_team": team,
        })
    return pd.DataFrame(rows)


def main() -> None:
    existing = pd.read_csv(OUT)
    print(f"existing prehistory: {len(existing)} rows, "
          f"{(existing['season'] == SEASON).sum()} in {SEASON}\n")

    frames = []
    missing_teams = []
    for team in TEAM_ABBREVS:
        df = scrape_team_2019_salaries(team)
        if df.empty:
            missing_teams.append(team)
        else:
            frames.append(df)
            print(f"  {team}: {len(df)} players")

    if not frames:
        raise SystemExit("No team pages parsed — refusing to write.")

    scraped = pd.concat(frames, ignore_index=True)
    aliases = _load_aliases()
    scraped["player_name_norm"] = scraped["player_name_norm"].replace(aliases)
    # A player traded mid-season appears on each team's page with the portion
    # that team paid; sum within (player, season) to the season cap hit, the
    # same convention backfill_prehistory_salaries.py uses. Keep the first
    # player_url seen.
    url_map = (scraped.dropna(subset=["player_url"])
               .drop_duplicates("player_name_norm")
               .set_index("player_name_norm")["player_url"].to_dict())
    agg = (scraped.groupby(["player_name_norm", "season"], as_index=False)
           ["salary"].sum())
    agg["player_url"] = agg["player_name_norm"].map(url_map).fillna("")

    # ---- extend-only reconciliation against existing rows ----
    exist_key = existing.set_index(["player_name_norm", "season"])["salary"].to_dict()
    conflicts, added = [], []
    new_rows = []
    for _, r in agg.iterrows():
        k = (r["player_name_norm"], int(r["season"]))
        if k in exist_key:
            old = int(exist_key[k])
            if abs(int(r["salary"]) - old) > CONFLICT_TOL:
                conflicts.append((k[0], old, int(r["salary"])))
            # keep existing byte-for-byte; do not append
        else:
            new_rows.append({
                "player_name_norm": r["player_name_norm"],
                "player_url": r["player_url"],
                "season": int(r["season"]),
                "salary": int(r["salary"]),
            })
            added.append(r["player_name_norm"])

    print(f"\nteams parsed: {len(TEAM_ABBREVS) - len(missing_teams)}/30"
          f"{'  MISSING: ' + ','.join(missing_teams) if missing_teams else ''}")
    print(f"scraped distinct 2018 players: {agg['player_name_norm'].nunique()}")
    print(f"new players added: {len(added)}")
    print(f"conflicts (existing kept, >|${CONFLICT_TOL:,}| apart): {len(conflicts)}")
    for name, old, new in sorted(conflicts):
        print(f"    CONFLICT {name}: existing ${old:,} vs scraped ${new:,} "
              f"(Δ${new - old:+,})")

    if missing_teams:
        print("\n  NOTE: a team stayed down; rerun to fetch it — existing rows "
              "are intact, only new players from that team are absent.")

    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    # match the existing file's ordering (player, season) so extended rows
    # insert alphabetically and the pre-existing rows keep their positions —
    # a clean, append-shaped diff rather than a full reshuffle
    out = out.sort_values(["player_name_norm", "season"],
                          kind="stable").reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.name}: {len(out)} rows "
          f"({(out['season'] == SEASON).sum()} in {SEASON}, "
          f"{out['player_name_norm'].nunique()} players total)")


if __name__ == "__main__":
    main()
