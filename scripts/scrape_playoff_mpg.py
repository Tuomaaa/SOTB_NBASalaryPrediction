"""Scrape BBRef playoff per-game tables into data/processed/playoff_mpg.csv.

Source of the `playoff_mpg_diff` feature (v8.6x): a player's playoff minutes per
game, which the model differences against his regular-season `mpg`. Rows with no
playoff appearance are absent here and fill to a zero difference downstream —
see `src/features/playoff_minutes.py`.

Season convention (verified against three known rows before it was trusted):
BBRef's page year N is the season that STARTS in year N-1, and the dataset's
`season = N` row carries exactly those stats (D'Angelo Russell 2023 -> 54 GP /
32.9 mpg, his 2022-23 Minnesota split; Tatum 2022 -> 76 GP / 35.9 mpg). So the
playoff page for a dataset row with season N is playoffs/NBA_{N}_per_game.html.

Two traps this script handles, both of which silently zero-fill real rows:
  - BBRef marks Hall-of-Fame players with a trailing asterisk ("Dwight Howard*"),
    which misses the name join. Stripped in `_normalize_name`.
  - The per-game table is sometimes buried inside an HTML comment. Parsed either
    way in `_parse`.

Discipline (CLAUDE.md):
  - A FAILED FETCH DEGRADES TO STALE, NEVER TO MISSING. A season whose page
    403s keeps whatever rows the existing CSV already holds for it; the script
    refuses to write if that would drop a season entirely.
  - Everything goes through the cached, rate-limited fetch_html. All eight
    seasons are already in data/raw/html_cache, so a rerun is offline and free.

Rerun-safe and idempotent given the cache.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup, Comment

from config import PROCESSED_DIR
from src.scraping.utils import fetch_html

SEASONS = list(range(2019, 2027))
URL = "https://www.basketball-reference.com/playoffs/NBA_{year}_per_game.html"
OUT = PROCESSED_DIR / "playoff_mpg.csv"
COLUMNS = ["player_name", "player_name_norm", "season", "po_team",
           "po_games", "po_mpg"]


def _normalize_name(name: str) -> str:
    """Same rule as src/features/build_dataset.py::_normalize_name.

    Plus one BBRef-specific strip the project's normalizer never needed: the
    trailing Hall-of-Fame asterisk. Without it Dwight Howard 2020/2021 and
    Carmelo Anthony miss the join and read as "did not play in the playoffs".
    """
    name = re.sub(r"\*+$", "", str(name).strip()).strip()
    name = name.lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _cell(row, *stats):
    """First non-empty cell among `stats`, by BBRef's data-stat attribute."""
    for stat in stats:
        el = row.find(["td", "th"], {"data-stat": stat})
        if el is not None:
            txt = el.get_text(strip=True)
            if txt:
                return txt
    return None


def _parse(html: str, year: int) -> pd.DataFrame:
    """One row per (player, season) with playoff games and minutes per game."""
    soup = BeautifulSoup(html, "html.parser")

    def find_table(node):
        for t in node.find_all("table"):
            if "per_game" in (t.get("id", "") or ""):
                return t
        return None

    table = find_table(soup)
    if table is None:  # BBRef sometimes buries secondary tables in comments
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            table = find_table(BeautifulSoup(str(c), "html.parser"))
            if table is not None:
                break
    if table is None:
        raise RuntimeError(f"no per_game table found for {year}")

    rows = []
    for tr in table.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        name = _cell(tr, "name_display", "player")
        if not name or name == "Player":
            continue
        mp = _cell(tr, "mp_per_g")
        g = _cell(tr, "g")
        team = _cell(tr, "team_name_abbr", "team_id")
        try:
            mp = float(mp) if mp is not None else None
            g = float(g) if g is not None else None
        except ValueError:
            continue
        if mp is None or g is None:
            continue
        rows.append({"player_name": name,
                     "player_name_norm": _normalize_name(name),
                     "season": year, "po_team": team,
                     "po_games": g, "po_mpg": mp})

    df = pd.DataFrame(rows)
    # A playoff run is one team, so a repeated (name, season) is a name
    # collision or a TOT-style aggregate; keep the most-minutes row.
    df["_tot"] = df["po_games"] * df["po_mpg"]
    df = df.sort_values("_tot", ascending=False).drop_duplicates(
        subset=["player_name_norm", "season"], keep="first")
    return df.drop(columns="_tot").sort_values("player_name_norm")


def _load_existing() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(OUT)


def scrape(seasons=SEASONS) -> pd.DataFrame:
    """Scrape every season, carrying a failed season's existing rows unchanged."""
    existing = _load_existing()
    frames, stale = [], []
    for year in seasons:
        try:
            df = _parse(fetch_html(URL.format(year=year)), year)
        except Exception as exc:  # 403 after retries, timeout, layout change
            kept = existing[existing["season"] == year]
            print(f"  [warn] {year}: fetch/parse failed, keeping "
                  f"{len(kept)} existing rows ({exc})")
            stale.append(year)
            if len(kept):
                frames.append(kept)
            continue
        print(f"  {year}: {len(df):4d} players, "
              f"mpg {df.po_mpg.min():.1f}-{df.po_mpg.max():.1f}, "
              f"games {df.po_games.min():.0f}-{df.po_games.max():.0f}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    # Degrade to stale, never to missing: a season that fails with nothing on
    # file would silently become "nobody played in the playoffs that year",
    # which is a zero-fill on every row of that season downstream.
    lost = [s for s in seasons if not (out["season"] == s).any()]
    if lost:
        raise SystemExit(f"refusing to write: no rows at all for season(s) {lost}")
    if stale:
        print(f"  carried stale rows for {stale}")
    return out


def _reconcile(out: pd.DataFrame) -> None:
    """Cross-check against the repo's other BBRef-derived playoff table.

    playoff_bpm_diff.csv carries po_minutes/po_games for 2020-2026 (2019 is
    absent), from an independent scrape, so it is a real second reading of the
    same quantity rather than a self-check.
    """
    ref_path = PROCESSED_DIR / "playoff_bpm_diff.csv"
    if not ref_path.exists():
        return
    ref = pd.read_csv(ref_path)
    ref = ref[ref["po_games"] > 0].copy()
    ref["ref_mpg"] = ref["po_minutes"] / ref["po_games"]
    m = ref.merge(out, on=["player_name_norm", "season"], how="left",
                  suffixes=("_ref", "_new"))
    ok = m[m["po_mpg"].notna()]
    d = (ok["ref_mpg"] - ok["po_mpg"]).abs()
    print(f"\nreconciliation vs playoff_bpm_diff.csv ({len(ref)} ref rows):")
    print(f"  matched into this scrape : {len(ok)}")
    print(f"  |mpg| agreement < 0.15   : {int((d < 0.15).sum())} / {len(ok)}")
    print(f"  games agreement exact    : "
          f"{int((ok['po_games_ref'] == ok['po_games_new']).sum())} / {len(ok)}")
    print(f"  max |mpg| disagreement   : {d.max():.3f}")


def main():
    out = scrape()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT}")
    print(out.groupby("season").size().to_string())
    _reconcile(out)


if __name__ == "__main__":
    main()
